"""Frozen relational ablations with the existing audio encoder and decoder."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch

from learned_possession.features import FeatureCache
from learned_possession.model import training_mask,fit_durations,near
from learned_possession.segment import DecoderParams,decode
from learned_possession.__main__ import read_labels,write_labels
from temporal_possession.audio import AudioCache,sha
from temporal_possession.network import known_runs,deterministic
from temporal_possession.experiment import metrics,save_json
from structured_possession.features import known_dead
from .data import GraphCache,SymmetricNormalizer
from .model import ARMS,Config,RelationalNet
from .engine import train_seed,infer_seed

LABELS=np.array(['H','A','D','X'])


def decode_scores(p,q,cache,durations,switch_cost=8):
    output=np.full(len(cache.x),'X',dtype='<U1')
    for _,first,last in cache.halves:
        output[first:last+1]=LABELS[decode(p[first:last+1],q[first:last+1],cache.fps,
                                          DecoderParams(switch_cost=switch_cost),durations)]
    output[known_dead(cache)]='D'
    return output


def fit_bundle(folder,gx,px,graph,cache,codes,selected,config,arm,seeds):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    durations,counts=fit_durations(codes,cache.fps,cache.halves)
    preprocess=folder.parent/'preprocessing'
    info=dict(schema=1,kind='relational_possession',arm=arm,config=asdict(config),seeds=list(seeds),
              global_features=gx.shape[1],pair_features=px.shape[2],node_features=graph.nodes.shape[2],
              durations=durations,duration_examples=counts,decoder=asdict(DecoderParams(switch_cost=8)),
              normalizer_hashes={name:sha(preprocess/name) for name in ('audio_normalizer.npz','symmetric.npz')},
              state_head='Two unverified latent substates per owner, aggregated into H/A; symmetric D',
              span_head='Auxiliary supervised interval distances; existing decoder unchanged')
    save_json(folder/'model.json',info)
    return info


def ensemble(folder,cache,audio,graph,device):
    folder=Path(folder)
    info=json.loads((folder/'model.json').read_text())
    if info.get('kind')!='relational_possession' or info['arm'] not in ARMS:
        raise ValueError('Unknown relational model')
    preprocess=folder.parent/'preprocessing'
    for name,expected in info['normalizer_hashes'].items():
        if sha(preprocess/name)!=expected:
            raise ValueError('Saved team-shared preprocessing changed')
    normalizer=SymmetricNormalizer.load(preprocess)
    gx,px=normalizer.transform(cache,audio)
    p,q=np.zeros((len(cache.x),3),np.float64),np.zeros(len(cache.x),np.float64)
    config=Config(**info['config'])
    for seed in info['seeds']:
        deterministic(seed)
        fit=json.loads((folder/f'seed_{seed}.json').read_text())
        checkpoint=folder/f'seed_{seed}.pt'
        if sha(checkpoint)!=fit['checkpoint_sha256']:
            raise ValueError('Relational checkpoint checksum mismatch')
        model=RelationalNet(info['global_features'],info['pair_features'],info['node_features'],config,info['arm']).to(device)
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        a,b=infer_seed(model,gx,px,graph,fit['boundary_positive_weight'],device)
        p+=a;q+=b
        del model
    return (p/len(info['seeds'])).astype(np.float32),(q/len(info['seeds'])).astype(np.float32),info


def train_group(folder,cache,audio,graph,codes,selected,config,seeds,device,fingerprints):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    normalizer=SymmetricNormalizer.fit(cache,audio,selected)
    normalizer.save(folder/'preprocessing')
    gx,px=normalizer.transform(cache,audio)
    predictions={}
    for arm in ARMS:
        print(f'Relational group {folder.name} arm {arm}',flush=True)
        dest=folder/arm
        info=fit_bundle(dest,gx,px,graph,cache,codes,selected,config,arm,seeds)
        info.update(fingerprints)
        save_json(dest/'model.json',info)
        for seed in seeds:
            model,fit=train_seed(gx,px,graph,codes,selected,config,arm,seed,device)
            checkpoint=dest/f'seed_{seed}.pt'
            torch.save(model.state_dict(),checkpoint)
            fit['checkpoint_sha256']=sha(checkpoint)
            save_json(dest/f'seed_{seed}.json',fit)
            del model
        p,q,_=ensemble(dest,cache,audio,graph,device)
        np.savez_compressed(dest/'scores.npz',probability=p,boundary=q)
        labels=decode_scores(p,q,cache,info['durations'])
        write_labels(dest/'predictions.csv',labels)
        predictions[arm]=labels
    return predictions


def calibrate(out,cache,codes,folds):
    """One fixed decoder parameter, selected on training-window OOF labels."""
    costs=(1.,2.,4.,8.)
    pooled={arm:{cost:np.full(len(codes),'X',dtype='<U1') for cost in costs} for arm in ARMS}
    for index,(held,_) in enumerate(folds):
        frames=np.flatnonzero(held!=3)
        first,stop=int(frames[0]),int(frames[-1])+1
        _,left,right=next(h for h in cache.halves if h[1]<=first and stop<=h[2]+1)
        for arm in ARMS:
            folder=out/'cv'/f'fold_{index}'/arm
            info=json.loads((folder/'model.json').read_text())
            with np.load(folder/'scores.npz',allow_pickle=False) as scores:
                p,q=scores['probability'],scores['boundary']
            for cost in costs:
                prediction=LABELS[decode(p[left:right+1],q[left:right+1],cache.fps,
                                        DecoderParams(switch_cost=cost),info['durations'])]
                pooled[arm][cost][first:stop]=prediction[first-left:stop-left]
        print(f'Relational decoder-grid fold {index+1}/{len(folds)} complete',flush=True)
    scores={arm:{cost:metrics(codes,labels,0) for cost,labels in grid.items()} for arm,grid in pooled.items()}
    chosen={arm:max(costs,key=lambda cost:scores[arm][cost]['spell_f1_mean']) for arm in ARMS}
    save_json(out/'decoder_selection.json',dict(cost_grid=costs,chosen=chosen,cv_scores=scores,
              criterion='pooled training-window OOF spellF1; ties choose lowest cost',
              main_validation_used_for_selection=False,selection_bias=True))
    outputs={}
    for arm,grid in pooled.items():
        for cost,labels in grid.items():
            write_labels(out/f'oof_{arm}_cost_{cost:g}.csv',labels)
        folder=out/'global'/arm
        info=json.loads((folder/'model.json').read_text())
        info['calibrated_switch_cost']=chosen[arm]
        info['calibration_source']='training-window OOF only; no neural parameter search'
        save_json(folder/'model.json',info)
        with np.load(folder/'scores.npz',allow_pickle=False) as scores_file:
            p,q=scores_file['probability'],scores_file['boundary']
        labels=decode_scores(p,q,cache,info['durations'],chosen[arm])
        write_labels(folder/'calibrated.csv',labels)
        outputs[arm+'_calibrated']=labels
    return outputs


def compare(args):
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError('No CUDA device available')
    deterministic(0)
    cache,audio,graph=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph)
    if cache.halves!=graph.halves or cache.fps!=graph.fps:
        raise ValueError('Graph and baseline observations do not share a timeline')
    train=read_labels(args.train_labels,len(cache.x))
    validation=read_labels(args.validation_labels,len(cache.x))
    audit=json.loads(Path(args.audit).read_text())
    if sha(args.train_labels)!=audit['train_labels_sha256'] or sha(args.validation_labels)!=audit['validation_labels_sha256']:
        raise ValueError('Label exports differ from the source audit')
    selected,valid=training_mask(cache,train,validation,60*cache.fps)
    codes=np.where(selected,train,3)
    config=Config(epochs=args.epochs)
    seeds=list(args.seeds)
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('Use distinct fixed seeds')
    fingerprints=dict(feature_sha256=sha(args.features),audio_sha256=sha(args.audio),graph_sha256=sha(args.graph))
    plan=dict(kind='team_symmetric_graph_memory_span_ablation',arms=ARMS,config=asdict(config),seeds=seeds,
              fingerprints=fingerprints,train_sha256=sha(args.train_labels),validation_sha256=sha(args.validation_labels),
              training_frames=int(selected.sum()),validation_frames=int(valid.sum()),annotation_audit=audit,
              decoder=asdict(DecoderParams(switch_cost=8)),decoder_frozen_for_all_arms=True,
              audio_encoder_changed=False,new_pretrained_backbones=False,graph_hz=cache.fps/5,
              supervision='Whole known training windows; 60s observation warmup; X/context never supervised',
              span_targets='Only verified interval ends; censored annotation-window edges masked',
              state_interpretation='Latent owner substates are not separately annotated control/transit labels',
              epochs_fixed=True,early_stopping=False,neural_parameter_search=False,
              decoder_calibration=bool(getattr(args,'calibrate',False)),decoder_cost_grid=[1,2,4,8],
              cv=bool(args.cross_validate),cv_seed=0,
              evaluation='Exploratory reused draft validation; whole-window CV; no full-match private GT')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    encoded=json.loads(json.dumps(plan))
    if (out/'experiment.json').exists() and json.loads((out/'experiment.json').read_text())!=encoded:
        raise ValueError('A different relational plan is already saved')
    save_json(out/'experiment.json',plan)
    print('Relational architecture plan frozen before fitting or scoring',flush=True)
    started=time.monotonic()
    folds=[]
    if args.cross_validate:
        for i,(first,stop) in enumerate(known_runs(codes!=3)):
            held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop]
            remainder=codes.copy();remainder[first:stop]=3
            chosen,_=training_mask(cache,remainder,held,60*cache.fps)
            chosen&=~near(valid,60*cache.fps)
            cv_codes=np.where(chosen,remainder,3)
            print(f'Relational CV fold {i+1}: held [{first},{stop}), train {chosen.sum()}',flush=True)
            predictions=train_group(out/'cv'/f'fold_{i}',cache,audio,graph,cv_codes,chosen,config,[0],args.device,fingerprints)
            folds.append((held,predictions))
    outputs=train_group(out/'global',cache,audio,graph,codes,selected,config,seeds,args.device,fingerprints)
    if getattr(args,'calibrate',False):
        if not folds:
            raise ValueError('Decoder selection requires held-out training windows')
        outputs.update(calibrate(out,cache,codes,folds))
    # Scoring is deferred until all folds and all final models finish.
    scores={name:metrics(validation,labels,args.bootstrap) for name,labels in outputs.items()}
    for path in args.compare:
        scores[str(path)]=metrics(validation,LABELS[read_labels(path,len(cache.x))],args.bootstrap)
    cv=[];pooled={arm:np.full(len(codes),'X',dtype='<U1') for arm in ARMS}
    for held,predictions in folds:
        cv.append({arm:metrics(held,labels,0) for arm,labels in predictions.items()})
        mask=held!=3
        for arm in ARMS:
            pooled[arm][mask]=predictions[arm][mask]
    oof={arm:metrics(codes,labels,args.bootstrap) for arm,labels in pooled.items()} if folds else {}
    for arm,labels in pooled.items():
        if folds:
            write_labels(out/f'oof_{arm}.csv',labels)
    report=dict(experiment=plan,scores=scores,cv=cv,pooled_oof=oof,elapsed_seconds=time.monotonic()-started,
                known_dead_misses={arm:int(np.sum(labels[known_dead(cache)]!='D')) for arm,labels in outputs.items()})
    save_json(out/'report.json',report)
    for name,score in scores.items():
        print(name,'spellF1',round(score['spell_f1_mean'],4),'macroF1',round(score['macro_f1'],4),flush=True)
    for name,score in oof.items():
        print('OOF',name,'spellF1',round(score['spell_f1_mean'],4),'predictions',score['n_pred_spells'],flush=True)
    print('RELATIONAL ARCHITECTURE STUDY COMPLETE',flush=True)


def predict(args):
    info=json.loads((Path(args.model)/'model.json').read_text())
    for key,path in (('feature_sha256',args.features),('audio_sha256',args.audio),('graph_sha256',args.graph)):
        if info[key]!=sha(path):
            raise ValueError('Prediction observations changed from the saved relational model')
    cache,audio,graph=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph)
    p,q,_=ensemble(args.model,cache,audio,graph,args.device)
    cost=info['calibrated_switch_cost'] if getattr(args,'calibrated',False) else 8
    write_labels(args.out,decode_scores(p,q,cache,info['durations'],cost))
    print('Relational regenerated SHA-256',sha(args.out),flush=True)
