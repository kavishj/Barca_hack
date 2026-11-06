"""Predeclared training-window selection, followed by one main evaluation."""
from pathlib import Path
import json
import shutil
import time
import joblib
import numpy as np
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask,near
from learned_possession.segment import decode,DecoderParams
from learned_possession.__main__ import read_labels,write_labels
from temporal_possession.network import Normalizer,known_runs
from temporal_possession.audio import AudioCache,sha
from temporal_possession.experiment import metrics,save_json
from structured_possession.features import known_dead
from .features import build
from .model import fit,predict,Config

LABELS=np.array(['H','A','D','X'])
VARIANTS=(('graph',8.),)+tuple((arm,cost) for arm in ('trees','blend') for cost in (1.,2.,4.,8.))


def key(arm,cost):return arm+'__'+str(int(cost))


def features_for(base,rich,audio,preprocessing):
    normalizer=Normalizer.load(Path(preprocessing)/'audio_normalizer.npz')
    raw=normalizer.raw(base,audio)
    return build(rich,raw[:,len(base.names):],normalizer.names[len(base.names):])


def decoded(p,q,cache,cost,durations,half=None):
    labels=np.full(len(cache.x),'X',dtype='<U1')
    for h,first,last in cache.halves:
        if half is None or h==half:
            labels[first:last+1]=LABELS[decode(p[first:last+1],q[first:last+1],cache.fps,
                DecoderParams(switch_cost=cost),durations)]
    forced=known_dead(cache)&(labels!='X');labels[forced]='D'
    return labels


def scores(folder,source,base,rich,audio,codes,threads):
    folder=Path(folder);source=Path(source);folder.mkdir(parents=True,exist_ok=True)
    prep=folder/'preprocessing';shutil.copytree(source/'preprocessing',prep,dirs_exist_ok=True)
    features=features_for(base,rich,audio,prep)
    model=fit(features,codes,threads=threads)
    joblib.dump(model,folder/'trees.joblib')
    p,q=predict(model,features,threads)
    np.savez_compressed(folder/'trees_scores.npz',probability=p,boundary=q)
    with np.load(source/'graph_tcn/scores.npz',allow_pickle=False) as d:gp,gq=d['probability'],d['boundary']
    np.savez_compressed(folder/'graph_scores.npz',probability=gp,boundary=gq)
    info=json.loads((source/'graph_tcn/model.json').read_text())
    save_json(folder/'model.json',dict(kind='symmetric_factorized_trees',config=model['config'],counts=model['counts'],
        durations=info['durations'],model_sha256=sha(folder/'trees.joblib'),graph_scores_sha256=sha(folder/'graph_scores.npz'),
        preprocessing_sha256={p.name:sha(p) for p in prep.glob('*.npz')},feature_names=features.names))
    return (gp,gq),(p,q),info['durations']


def combine(arm,graph,trees):
    if arm=='graph':return graph
    if arm=='trees':return trees
    if arm=='blend':return tuple((g+t)/2 for g,t in zip(graph,trees))
    raise ValueError('Invalid boosted arm')


def compare(args):
    source,out=Path(args.source_study),Path(args.out);out.mkdir(parents=True,exist_ok=True)
    base,rich,audio=FeatureCache.load(args.features),FeatureCache.load(args.rich_features),AudioCache.load(args.audio)
    if len(base.x)!=len(rich.x) or base.halves!=rich.halves or base.fps!=rich.fps:raise ValueError('Rich timeline mismatch')
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    original=json.loads((source/'experiment.json').read_text())
    if sha(args.train_labels)!=original['train_sha256'] or sha(args.validation_labels)!=original['validation_sha256']:
        raise ValueError('References changed')
    selected,valid=training_mask(base,train,validation,60*base.fps);codes=np.where(selected,train,3)
    plan=dict(kind='symmetric_factorized_boosted_trees',config=vars(Config()),variants=VARIANTS,
        fingerprints={n:sha(getattr(args,n)) for n in ('features','rich_features','audio','train_labels','validation_labels')},
        train_frames=int(selected.sum()),validation_frames=int(valid.sum()),validation_used_for_selection=False,
        selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 11')
    for n in ('features','audio'):
        if plan['fingerprints'][n]!=original['fingerprints'][{'features':'feature_sha256','audio':'audio_sha256'}[n]]:
            raise ValueError('Original observations changed')
    if (out/'experiment.json').exists():raise ValueError('Use a fresh directory; do not silently reuse a different study')
    save_json(out/'experiment.json',plan);started=time.monotonic()
    pooled={key(*v):np.full(len(codes),'X',dtype='<U1') for v in VARIANTS};fold_scores=[]
    for index,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop]
        remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(base,remainder,held,60*base.fps);chosen&=~near(valid,60*base.fps)
        cv_codes=np.where(chosen,remainder,3);folder=out/'cv'/f'fold_{index}'
        print('BOOSTED fit fold',index,flush=True)
        graph,trees,durations=scores(folder,source/'cv'/f'fold_{index}',base,rich,audio,cv_codes,args.threads)
        half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        for arm,cost in VARIANTS:
            labels=decoded(*combine(arm,graph,trees),base,cost,durations,half)
            name=key(arm,cost);pooled[name][first:stop]=labels[first:stop]
            write_labels(folder/(name+'.csv'),labels);result[name]=metrics(held,labels,0)
        fold_scores.append(result);print('BOOSTED OOF fold',index,'complete',flush=True)
    oof={name:metrics(codes,labels,0) for name,labels in pooled.items()}
    winner=max(oof,key=lambda name:oof[name]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True,tie_order=[key(*v) for v in VARIANTS])
    save_json(out/'decision.json',decision)
    for name,labels in pooled.items():write_labels(out/('oof_'+name+'.csv'),labels)
    print('BOOSTED selected',winner,flush=True)
    graph,trees,durations=scores(out/'global',source/'global',base,rich,audio,codes,args.threads)
    arm,cost=winner.split('__');prediction=decoded(*combine(arm,graph,trees),base,float(cost),durations)
    write_labels(out/'selected_predictions.csv',prediction)
    info=json.loads((out/'global/model.json').read_text());info.update(selected=winner,input_sha256=plan['fingerprints'])
    save_json(out/'global/model.json',info)
    report=dict(experiment=plan,decision=decision,fold_scores=fold_scores,selected=winner,
        selected_main=metrics(validation,prediction,args.bootstrap),elapsed_seconds=time.monotonic()-started)
    save_json(out/'report.json',report)
    for name,result in oof.items():print('BOOSTED OOF',name,result['spell_f1_mean'],result['n_pred_spells'],flush=True)
    print('BOOSTED MAIN',winner,report['selected_main']['spell_f1_mean'],report['selected_main']['macro_f1'],flush=True)


def regenerate(args):
    folder=Path(args.model);info=json.loads((folder/'model.json').read_text())
    for name in ('features','rich_features','audio'):
        if sha(getattr(args,name))!=info['input_sha256'][name]:raise ValueError('Inference input changed')
    for name,digest in info['preprocessing_sha256'].items():
        if sha(folder/'preprocessing'/name)!=digest:raise ValueError('Saved preprocessing changed')
    if sha(folder/'trees.joblib')!=info['model_sha256'] or sha(folder/'graph_scores.npz')!=info['graph_scores_sha256']:
        raise ValueError('Model or frozen graph score cache changed')
    base,rich,audio=FeatureCache.load(args.features),FeatureCache.load(args.rich_features),AudioCache.load(args.audio)
    features=features_for(base,rich,audio,folder/'preprocessing')
    trees=predict(joblib.load(folder/'trees.joblib'),features,args.threads)
    with np.load(folder/'graph_scores.npz',allow_pickle=False) as d:graph=(d['probability'],d['boundary'])
    arm,cost=info['selected'].split('__')
    write_labels(args.out,decoded(*combine(arm,graph,trees),base,float(cost),info['durations']))
