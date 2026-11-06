"""Controlled frozen-video addition to the saved winning graph experiment."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from learned_possession.features import FeatureCache
from learned_possession.__main__ import read_labels,write_labels
from learned_possession.model import training_mask,fit_durations,near
from learned_possession.segment import DecoderParams,decode
from temporal_possession.audio import AudioCache,sha
from temporal_possession.network import deterministic,known_runs
from temporal_possession.experiment import metrics,save_json
from relational_possession.data import GraphCache,SymmetricNormalizer
from relational_possession.model import Config
from relational_possession.engine import infer_seed
from relational_possession.experiment import decode_scores
from .cache import VisualCache,VisualNormalizer
from .model import VisionNet,VisualGraph
from .engine import train_seed

LABELS=np.array(['H','A','D','X'])


def observations(args):
    cache,audio,graph,vision=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph),VisualCache.load(args.vision)
    if cache.halves!=vision.halves or cache.fps!=vision.fps or graph.halves!=cache.halves:
        raise ValueError('Visual and graph observations do not share a timeline')
    return cache,audio,graph,vision


def ensemble(folder,cache,audio,graph,vision,device):
    folder=Path(folder);info=json.loads((folder/'model.json').read_text())
    preprocessing=folder.parent/'preprocessing'
    for name,digest in info['normalizer_hashes'].items():
        if sha(preprocessing/name)!=digest:raise ValueError('Frozen visual/base preprocessing changed')
    base=SymmetricNormalizer.load(preprocessing)
    gx,px=base.transform(cache,audio)
    vg,vp=VisualNormalizer.load(preprocessing/'visual_normalizer.npz').transform(vision,len(cache.x))
    joined=VisualGraph(graph,vg,vp)
    p=np.zeros((len(cache.x),3),np.float64);q=np.zeros(len(cache.x),np.float64)
    config=Config(**info['config'])
    for seed in info['seeds']:
        deterministic(seed)
        fit=json.loads((folder/f'seed_{seed}.json').read_text())
        if sha(folder/f'seed_{seed}.pt')!=fit['checkpoint_sha256']:raise ValueError('Visual model checkpoint changed')
        model=VisionNet(info['global_features'],info['pair_features'],info['node_features'],config,info['vision_features']).to(device)
        model.load_state_dict(torch.load(folder/f'seed_{seed}.pt',map_location=device,weights_only=True))
        a,b=infer_seed(model,gx,px,joined,fit['boundary_positive_weight'],device)
        p+=a;q+=b
        del model
    return (p/len(info['seeds'])).astype(np.float32),(q/len(info['seeds'])).astype(np.float32),info


def fit_group(folder,control,cache,audio,graph,vision,codes,selected,config,seeds,device,fingerprints):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    dest=folder/'graph_vision';dest.mkdir(exist_ok=True)
    prep=folder/'preprocessing';prep.mkdir(exist_ok=True)
    before=Path(control)/'preprocessing'
    for name in ('audio_normalizer.npz','symmetric.npz'):shutil.copy2(before/name,prep/name)
    normalizer=SymmetricNormalizer.load(prep)
    gx,px=normalizer.transform(cache,audio)
    visual=VisualNormalizer.fit(vision,selected,16);visual.save(prep/'visual_normalizer.npz')
    vg,vp=visual.transform(vision,len(cache.x));joined=VisualGraph(graph,vg,vp)
    reference=json.loads((Path(control)/'graph_tcn/model.json').read_text())
    durations,counts=fit_durations(codes,cache.fps,cache.halves)
    if json.loads(json.dumps(durations))!=reference['durations']:raise ValueError('Training windows differ from original graph control')
    info=dict(kind='vision_possession',config=asdict(config),seeds=list(seeds),global_features=gx.shape[1],
        pair_features=px.shape[2],node_features=graph.nodes.shape[2],vision_features=vg.shape[1],
        durations=durations,duration_examples=counts,decoder=asdict(DecoderParams(switch_cost=8)),
        normalizer_hashes={name:sha(prep/name) for name in ('audio_normalizer.npz','symmetric.npz','visual_normalizer.npz')},
        cached_encoder_provenance=vision.provenance,control_preprocessing_unchanged=True,**fingerprints)
    save_json(dest/'model.json',info)
    for seed in seeds:
        model,fit=train_seed(gx,px,joined,codes,selected,config,seed,device)
        torch.save(model.state_dict(),dest/f'seed_{seed}.pt')
        fit['checkpoint_sha256']=sha(dest/f'seed_{seed}.pt');save_json(dest/f'seed_{seed}.json',fit)
        del model
    p,q,_=ensemble(dest,cache,audio,graph,vision,device)
    np.savez_compressed(dest/'scores.npz',probability=p,boundary=q)
    prediction=decode_scores(p,q,cache,durations,8)
    write_labels(dest/'predictions.csv',prediction)
    shutil.copy2(Path(control)/'graph_tcn/predictions.csv',folder/'control.csv')
    return prediction


def score_and_calibrate(out,source,cache,codes,validation,folds,bootstrap):
    costs=(1.,2.,4.,8.)
    pooled={cost:np.full(len(codes),'X',dtype='<U1') for cost in costs}
    fold_scores=[]
    for index,(held,_) in enumerate(folds):
        take=np.flatnonzero(held!=3);first,stop=int(take[0]),int(take[-1])+1
        _,left,right=next(h for h in cache.halves if h[1]<=first and stop<=h[2]+1)
        folder=out/'cv'/f'fold_{index}'
        info=json.loads((folder/'graph_vision/model.json').read_text())
        with np.load(folder/'graph_vision/scores.npz',allow_pickle=False) as d:p,q=d['probability'],d['boundary']
        local={}
        for cost in costs:
            decoded=LABELS[decode(p[left:right+1],q[left:right+1],cache.fps,
                DecoderParams(switch_cost=cost),info['durations'])]
            pooled[cost][first:stop]=decoded[first-left:stop-left]
            local[str(cost)]=metrics(held,np.r_[np.full(left,'X'),decoded,np.full(len(codes)-right-1,'X')],0)
        fold_scores.append(local)
        print('Vision decoder-grid fold',index+1,'complete',flush=True)
    grid={str(cost):metrics(codes,labels,0) for cost,labels in pooled.items()}
    chosen=max(costs,key=lambda c:grid[str(c)]['spell_f1_mean'])
    save_json(out/'decoder_selection.json',dict(grid=grid,chosen=chosen,costs=costs,
        validation_used_for_selection=False,selection_bias=True))
    for cost,labels in pooled.items():write_labels(out/f'oof_vision_cost_{cost:g}.csv',labels)
    model=out/'global/graph_vision'
    info=json.loads((model/'model.json').read_text());info['calibrated_switch_cost']=chosen
    save_json(model/'model.json',info)
    with np.load(model/'scores.npz',allow_pickle=False) as d:p,q=d['probability'],d['boundary']
    selected=decode_scores(p,q,cache,info['durations'],chosen)
    write_labels(model/'calibrated.csv',selected)
    control=LABELS[read_labels(source/'global/graph_tcn/predictions.csv',len(codes))]
    raw=LABELS[read_labels(model/'predictions.csv',len(codes))]
    scores={key:metrics(validation,labels,bootstrap) for key,labels in (
        ('graph_control',control),('graph_vision_cost_8',raw),('graph_vision_calibrated',selected))}
    old_oof=LABELS[read_labels(source/'oof_graph_tcn.csv',len(codes))]
    cv=dict(graph_control=metrics(codes,old_oof,bootstrap),graph_vision=grid[str(chosen)])
    for key,s in scores.items():print('VISION RESULT',key,'spellF1',s['spell_f1_mean'],'macroF1',s['macro_f1'],flush=True)
    print('VISION OOF',cv['graph_control']['spell_f1_mean'],'->',cv['graph_vision']['spell_f1_mean'],flush=True)
    return dict(scores=scores,pooled_oof=cv,grid=grid,fold_grid=fold_scores,selected_cost=chosen)


def compare(args):
    cache,audio,graph,vision=observations(args)
    source=Path(args.source_study);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    previous=json.loads((source/'experiment.json').read_text())
    train,validation=read_labels(args.train_labels,len(cache.x)),read_labels(args.validation_labels,len(cache.x))
    if sha(args.train_labels)!=previous['train_sha256'] or sha(args.validation_labels)!=previous['validation_sha256']:
        raise ValueError('References differ from the original graph comparison')
    fingerprints={name:sha(path) for name,path in (('feature_sha256',args.features),('audio_sha256',args.audio),
                                                ('graph_sha256',args.graph),('vision_sha256',args.vision))}
    for name in ('feature_sha256','audio_sha256','graph_sha256'):
        if fingerprints[name]!=previous['fingerprints'][name]:raise ValueError('Base graph/audio observations changed')
    selected,valid=training_mask(cache,train,validation,60*cache.fps);codes=np.where(selected,train,3)
    config=Config(epochs=args.epochs)
    if config.epochs!=previous['config']['epochs']:raise ValueError('Use the original graph training epoch budget')
    plan=dict(kind='frozen_video_addition_to_graph_tcn',config=asdict(config),seeds=list(args.seeds),
        fingerprints=fingerprints,train_sha256=sha(args.train_labels),validation_sha256=sha(args.validation_labels),
        reference_frames=int(valid.sum()),training_frames=int(selected.sum()),
        visual_encoder_frozen=True,audio_changed=False,memory_and_span_disabled=True,
        control='Saved original graph fits; same input normalization and optimization; zero-gated equivalence tested',
        full_match_visual_coverage=vision.provenance.get('full_match',False),
        decoder_costs=[1,2,4,8],selection='Pooled training-window OOF only; reused draft validation',
        source_study=str(source))
    save_json(out/'experiment.json',plan)
    print('VISION EXPERIMENT PLAN FROZEN before training and scoring',flush=True)
    started=time.monotonic();folds=[]
    for index,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop]
        remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(cache,remainder,held,60*cache.fps);chosen&=~near(valid,60*cache.fps)
        cv_codes=np.where(chosen,remainder,3)
        print('Vision CV fold',index+1,'held',first,stop,flush=True)
        prediction=fit_group(out/'cv'/f'fold_{index}',source/'cv'/f'fold_{index}',cache,audio,graph,
            vision,cv_codes,chosen,config,[0],args.device,fingerprints)
        folds.append((held,prediction))
    fit_group(out/'global',source/'global',cache,audio,graph,vision,codes,selected,config,args.seeds,args.device,fingerprints)
    report=score_and_calibrate(out,source,cache,codes,validation,folds,args.bootstrap)
    report.update(experiment=plan,elapsed_seconds=time.monotonic()-started)
    save_json(out/'report.json',report)
    print('VISION FUSION STUDY COMPLETE',flush=True)


def predict(args):
    info=json.loads((Path(args.model)/'model.json').read_text())
    for name,path in (('feature_sha256',args.features),('audio_sha256',args.audio),('graph_sha256',args.graph),('vision_sha256',args.vision)):
        if sha(path)!=info[name]:raise ValueError('Inference cache differs from saved visual model')
    cache,audio,graph,vision=observations(args)
    p,q,_=ensemble(args.model,cache,audio,graph,vision,args.device)
    cost=info['calibrated_switch_cost'] if args.calibrated else 8
    write_labels(args.out,decode_scores(p,q,cache,info['durations'],cost))
    print('VISION REGENERATED',sha(args.out),flush=True)
