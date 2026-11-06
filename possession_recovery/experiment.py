"""Four fixed graph/loss variants, OOF selection before one main evaluation."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from learned_possession.features import FeatureCache
from learned_possession.__main__ import read_labels,write_labels
from learned_possession.model import training_mask,near,boundary_targets
from learned_possession.segment import decode as fixed_decode,DecoderParams
from temporal_possession.audio import AudioCache,sha
from temporal_possession.network import deterministic,known_runs
from temporal_possession.experiment import metrics,save_json
from relational_possession.data import GraphCache,SymmetricNormalizer
from relational_possession.model import Config,RelationalNet
from relational_possession.engine import infer_seed
from structured_possession.features import known_dead
from .model import RecoveryNet
from .engine import train_seed
from .decoder import decode as adaptive_decode

ARMS=('graph_natural','graph_balanced','event_natural','event_balanced')
POLICIES=('fixed','adaptive')
LABELS=np.array(['H','A','D','X'])


def training_prior(codes):
    target,usable=boundary_targets(codes,3)
    prior=float(target[usable].mean())
    if not 0<prior<1:raise ValueError('Cannot fit a boundary prior from these training labels')
    return prior


def decode_scores(p,q,cache,info,policy,half_only=None):
    result=np.full(len(cache.x),'X',dtype='<U1')
    for half,first,last in cache.halves:
        if half_only is not None and half!=half_only:continue
        if policy=='fixed':
            decoded=fixed_decode(p[first:last+1],q[first:last+1],cache.fps,DecoderParams(switch_cost=8),info['durations'])
        elif policy=='adaptive':
            decoded=adaptive_decode(p[first:last+1],q[first:last+1],cache.fps,info['boundary_training_prior'])
        else:raise ValueError('Unknown recovery decoder policy')
        result[first:last+1]=LABELS[decoded]
    forced=known_dead(cache)
    if half_only is None:result[forced]='D'
    else:result[forced&(result!='X')]='D'
    return result


def ensemble(folder,cache,audio,graph,device):
    folder=Path(folder);info=json.loads((folder/'model.json').read_text())
    prep=folder.parent/'preprocessing'
    for name,digest in info['normalizer_hashes'].items():
        if sha(prep/name)!=digest:raise ValueError('Saved recovery preprocessing changed')
    gx,px=SymmetricNormalizer.load(prep).transform(cache,audio)
    p=np.zeros((len(cache.x),3),np.float64);q=np.zeros(len(cache.x),np.float64)
    config=Config(**info['config'])
    for seed in info['seeds']:
        deterministic(seed);fit=json.loads((folder/f'seed_{seed}.json').read_text())
        if sha(folder/f'seed_{seed}.pt')!=fit['checkpoint_sha256']:raise ValueError('Recovery checkpoint changed')
        if info['kind']=='relational_possession':
            model=RelationalNet(info['global_features'],info['pair_features'],15,config,'graph_tcn').to(device)
            observed=GraphCache(graph.frames,graph.half_ids,np.ascontiguousarray(graph.nodes[:,:,:15]),graph.roles,
                graph.valid,graph.positioned,graph.confidence,graph.halves,graph.fps)
        else:
            model=RecoveryNet(info['global_features'],info['pair_features'],graph.nodes.shape[2],config,info['history']).to(device)
            observed=graph
        model.load_state_dict(torch.load(folder/f'seed_{seed}.pt',map_location=device,weights_only=True))
        a,b=infer_seed(model,gx,px,observed,fit['boundary_positive_weight'],device)
        p+=a;q+=b;del model
    return (p/len(info['seeds'])).astype(np.float32),(q/len(info['seeds'])).astype(np.float32),info


def fit_group(folder,source,cache,audio,graph,codes,selected,config,seeds,device,fingerprints):
    folder=Path(folder);source=Path(source);folder.mkdir(parents=True,exist_ok=True)
    prep=folder/'preprocessing';prep.mkdir(exist_ok=True)
    for name in ('audio_normalizer.npz','symmetric.npz'):shutil.copy2(source/'preprocessing'/name,prep/name)
    gx,px=SymmetricNormalizer.load(prep).transform(cache,audio)
    reference=json.loads((source/'graph_tcn/model.json').read_text())
    prior=training_prior(codes)
    for arm in ARMS:
        dest=folder/arm;dest.mkdir(exist_ok=True)
        if arm=='graph_natural':
            for path in (source/'graph_tcn').glob('seed_*'):shutil.copy2(path,dest/path.name)
            info=dict(reference,**fingerprints,boundary_training_prior=prior,recovery_arm=arm)
            with np.load(source/'graph_tcn/scores.npz',allow_pickle=False) as data:p,q=data['probability'],data['boundary']
        else:
            history=arm.startswith('event');balanced=arm.endswith('balanced')
            info=dict(kind='possession_recovery',config=asdict(config),seeds=list(seeds),global_features=gx.shape[1],
                pair_features=px.shape[2],node_features=graph.nodes.shape[2],durations=reference['durations'],history=history,
                balanced=balanced,recovery_arm=arm,boundary_training_prior=prior,
                normalizer_hashes=reference['normalizer_hashes'],**fingerprints)
            save_json(dest/'model.json',info)
            for seed in seeds:
                checkpoint=dest/f'seed_{seed}.pt';fit_path=dest/f'seed_{seed}.json'
                if checkpoint.exists() and fit_path.exists():
                    if sha(checkpoint)!=json.loads(fit_path.read_text())['checkpoint_sha256']:raise ValueError('Resumed checkpoint changed')
                    continue
                print('Recovery fit',folder.name,arm,'seed',seed,flush=True)
                model,fit=train_seed(gx,px,graph,codes,selected,config,seed,device,history,balanced)
                torch.save(model.state_dict(),checkpoint);fit['checkpoint_sha256']=sha(checkpoint)
                save_json(fit_path,fit);del model
            p,q,_=ensemble(dest,cache,audio,graph,device)
        save_json(dest/'model.json',info)
        np.savez_compressed(dest/'scores.npz',probability=p,boundary=q)
        # Only observation-derived full-match scores are saved here. Evaluation follows all fits.
    return prior


def compare(args):
    source,out=Path(args.source_study),Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cache,audio,graph=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph)
    original=json.loads((source/'experiment.json').read_text())
    with np.load(args.graph,allow_pickle=False) as d:graph_metadata=json.loads(str(d['metadata']))
    if graph_metadata['base_graph_sha256']!=original['fingerprints']['graph_sha256']:raise ValueError('Contact graph is not based on the saved control observations')
    train,validation=read_labels(args.train_labels,len(cache.x)),read_labels(args.validation_labels,len(cache.x))
    if sha(args.train_labels)!=original['train_sha256'] or sha(args.validation_labels)!=original['validation_sha256']:
        raise ValueError('Recovery reference files changed')
    fingerprints=dict(feature_sha256=sha(args.features),audio_sha256=sha(args.audio),history_graph_sha256=sha(args.graph))
    for name in ('feature_sha256','audio_sha256'):
        if fingerprints[name]!=original['fingerprints'][name]:raise ValueError('Base audio/features changed')
    selected,valid=training_mask(cache,train,validation,60*cache.fps);codes=np.where(selected,train,3)
    config=Config(epochs=args.epochs);seeds=list(args.seeds)
    if json.loads(json.dumps(asdict(config)))!=original['config'] or seeds!=original['seeds']:
        raise ValueError('Use the original graph configuration and seed budget')
    plan=dict(kind='contact_and_spell_balanced_recovery',arms=ARMS,decoders=POLICIES,config=asdict(config),seeds=seeds,
        fingerprints=fingerprints,train_sha256=sha(args.train_labels),validation_sha256=sha(args.validation_labels),
        training_frames=int(selected.sum()),validation_frames=int(valid.sum()),main_validation_uses='Score selected OOF variant only',
        selection='Maximum pooled training-window OOF official spellF1; first listed arm/decoder breaks ties',
        validation_reused=True,selection_bias=True,protocol='F1_TARGET.md experiment 10')
    canonical=json.loads(json.dumps(plan))
    if (out/'experiment.json').exists() and json.loads((out/'experiment.json').read_text())!=canonical:
        raise ValueError('Recovery output directory contains a different fixed plan')
    save_json(out/'experiment.json',plan)
    print('RECOVERY PLAN FROZEN before fitting or main scoring',flush=True);started=time.monotonic()
    folds=[]
    for index,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop]
        remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(cache,remainder,held,60*cache.fps);chosen&=~near(valid,60*cache.fps)
        cv_codes=np.where(chosen,remainder,3)
        print('Recovery CV fold',index+1,'held',first,stop,flush=True)
        fit_group(out/'cv'/f'fold_{index}',source/'cv'/f'fold_{index}',cache,audio,graph,cv_codes,chosen,config,[0],args.device,fingerprints)
        folds.append((first,stop,held))
    fit_group(out/'global',source/'global',cache,audio,graph,codes,selected,config,seeds,args.device,fingerprints)
    pooled={(arm,policy):np.full(len(codes),'X',dtype='<U1') for arm in ARMS for policy in POLICIES}
    fold_scores=[]
    for index,(first,stop,held) in enumerate(folds):
        half=next(h[0] for h in cache.halves if h[1]<=first and stop<=h[2]+1);scores={}
        for arm in ARMS:
            folder=out/'cv'/f'fold_{index}'/arm;info=json.loads((folder/'model.json').read_text())
            with np.load(folder/'scores.npz',allow_pickle=False) as d:p,q=d['probability'],d['boundary']
            for policy in POLICIES:
                labels=decode_scores(p,q,cache,info,policy,half_only=half)
                write_labels(folder/f'{policy}.csv',labels)
                pooled[arm,policy][first:stop]=labels[first:stop]
                scores[arm+'__'+policy]=metrics(held,labels,0)
        fold_scores.append(scores);print('Recovery OOF decoding fold',index+1,'complete',flush=True)
    scores={arm+'__'+policy:metrics(codes,labels,0) for (arm,policy),labels in pooled.items()}
    winner=max(scores,key=lambda key:scores[key]['spell_f1_mean']);arm,policy=winner.split('__')
    decision=dict(selected=winner,oof_scores=scores,validation_used=False,tie_policy=plan['selection'],selection_bias=True)
    save_json(out/'decision.json',decision)
    for (a,p),labels in pooled.items():write_labels(out/f'oof_{a}__{p}.csv',labels)
    chosen_model=out/'global'/arm;info=json.loads((chosen_model/'model.json').read_text());info['recovery_decoder']=policy
    save_json(chosen_model/'model.json',info)
    with np.load(chosen_model/'scores.npz',allow_pickle=False) as d:p,q=d['probability'],d['boundary']
    prediction=decode_scores(p,q,cache,info,policy)
    write_labels(out/'selected_predictions.csv',prediction)
    deployment=out/'deployment/model';deployment.mkdir(parents=True,exist_ok=True)
    for path in chosen_model.glob('seed_*'):shutil.copy2(path,deployment/path.name)
    save_json(deployment/'model.json',info)
    shutil.copytree(out/'global/preprocessing',deployment.parent/'preprocessing',dirs_exist_ok=True)
    main=metrics(validation,prediction,args.bootstrap)
    control=LABELS[read_labels(source/'global/graph_tcn/predictions.csv',len(codes))]
    report=dict(experiment=plan,oof_scores=scores,fold_scores=fold_scores,selected=winner,
        selected_main=main,control_main=metrics(validation,control,args.bootstrap),elapsed_seconds=time.monotonic()-started)
    save_json(out/'report.json',report)
    for name,score in scores.items():print('RECOVERY OOF',name,score['spell_f1_mean'],score['n_pred_spells'],flush=True)
    print('RECOVERY SELECTED',winner,'spellF1',main['spell_f1_mean'],'macroF1',main['macro_f1'],flush=True)
    print('RECOVERY STUDY COMPLETE',flush=True)


def predict(args):
    info=json.loads((Path(args.model)/'model.json').read_text())
    for key,path in (('feature_sha256',args.features),('audio_sha256',args.audio),('history_graph_sha256',args.graph)):
        if sha(path)!=info[key]:raise ValueError('Recovery inference cache changed')
    cache,audio,graph=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph)
    p,q,_=ensemble(args.model,cache,audio,graph,args.device)
    write_labels(args.out,decode_scores(p,q,cache,info,info['recovery_decoder']))
    print('RECOVERY REGENERATED',sha(args.out),flush=True)
