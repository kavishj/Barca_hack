"""Fixed pitch-reflection experiment, training-only decoder selection."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask,near
from learned_possession.__main__ import read_labels,write_labels
from temporal_possession.network import known_runs,deterministic
from temporal_possession.audio import AudioCache,sha
from temporal_possession.experiment import metrics,save_json
from relational_possession.data import GraphCache,SymmetricNormalizer
from relational_possession.engine import train_seed,infer_seed
from boosted_possession.experiment import decoded,combine
from relational_possession.model import Config,RelationalNet
from .augmentation import Reflections

VARIANTS=(('graph',8.),('blend',2.))+tuple(('reflected',c) for c in (1.,2.,4.,8.))
def key(arm,cost):return arm+'__'+str(int(cost))


def ensemble(folder,base,audio,graph,device):
    folder=Path(folder);info=json.loads((folder/'model.json').read_text())
    for name,digest in info['preprocessing_sha256'].items():
        if sha(folder/'preprocessing'/name)!=digest:raise ValueError('Reflection preprocessing changed')
    gx,px=SymmetricNormalizer.load(folder/'preprocessing').transform(base,audio)
    p=np.zeros((len(base.x),3),np.float64);q=np.zeros(len(base.x),np.float64)
    for seed in info['seeds']:
        fit=json.loads((folder/f'seed_{seed}.json').read_text())
        if sha(folder/f'seed_{seed}.pt')!=fit['checkpoint_sha256']:raise ValueError('Reflection checkpoint changed')
        deterministic(seed);model=RelationalNet(gx.shape[1],px.shape[2],15,Config(**info['config']),'graph_tcn').to(device)
        model.load_state_dict(torch.load(folder/f'seed_{seed}.pt',map_location=device,weights_only=True))
        a,b=infer_seed(model,gx,px,graph,fit['boundary_positive_weight'],device)
        p+=a;q+=b;del model
    return (p/len(info['seeds'])).astype(np.float32),(q/len(info['seeds'])).astype(np.float32)


def fit_group(folder,source,base,audio,graph,codes,selected,seeds,epochs,device):
    folder=Path(folder);source=Path(source);folder.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source/'preprocessing',folder/'preprocessing',dirs_exist_ok=True)
    gx,px=SymmetricNormalizer.load(folder/'preprocessing').transform(base,audio);config=Config(epochs=epochs)
    prior=json.loads((source/'graph_tcn/model.json').read_text())
    info=dict(kind='reflected_possession',config=asdict(config),seeds=list(seeds),durations=prior['durations'],
              preprocessing_sha256={p.name:sha(p) for p in (folder/'preprocessing').glob('*.npz')})
    save_json(folder/'model.json',info)
    for seed in seeds:
        print('REFLECTED fit',folder.name,'seed',seed,flush=True)
        model,fit=train_seed(gx,px,graph,codes,selected,config,'graph_tcn',seed,device,model_factory=RelationalNet,observation_transform=Reflections(SymmetricNormalizer.load(folder/'preprocessing'),seed))
        checkpoint=folder/f'seed_{seed}.pt';torch.save(model.state_dict(),checkpoint)
        fit['checkpoint_sha256']=sha(checkpoint);save_json(folder/f'seed_{seed}.json',fit);del model
    p,q=ensemble(folder,base,audio,graph,device)
    np.savez_compressed(folder/'scores.npz',probability=p,boundary=q)
    return p,q,info['durations']


def group_scores(folder,source,boosted):
    result={}
    for name,path in (('graph',Path(source)/'graph_tcn/scores.npz'),('trees',Path(boosted)/'trees_scores.npz'),('reflected',Path(folder)/'scores.npz')):
        with np.load(path,allow_pickle=False) as d:result[name]=(d['probability'],d['boundary'])
    result['blend']=combine('blend',result['graph'],result['trees'])
    return result


def compare(args):
    source,boosted,out=Path(args.source_study),Path(args.boosted_study),Path(args.out)
    if out.exists():raise ValueError('Use a fresh coarse study directory')
    out.mkdir(parents=True)
    base,audio,graph=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph)
    original=json.loads((source/'experiment.json').read_text());trees=json.loads((boosted/'experiment.json').read_text())
    fingerprints={name:sha(getattr(args,name)) for name in ('features','audio','graph','train_labels','validation_labels')}
    expected=dict(features=original['fingerprints']['feature_sha256'],audio=original['fingerprints']['audio_sha256'],
        graph=original['fingerprints']['graph_sha256'],train_labels=original['train_sha256'],validation_labels=original['validation_sha256'])
    if fingerprints!=expected:raise ValueError('Original observations or labels changed')
    if any(trees['fingerprints'][n]!=fingerprints[n] for n in ('features','audio','train_labels','validation_labels')):
        raise ValueError('Boosted control is based on different inputs')
    for half,first,last in graph.halves:
        frame=graph.frames[graph.half_ids==half]
        if not np.array_equal(frame,np.unique(np.r_[np.arange(first,last+1,5),last])):raise ValueError('Expected original 5 Hz graph')
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    selected,valid=training_mask(base,train,validation,60*base.fps);codes=np.where(selected,train,3)
    plan=dict(kind='reflected_graph_temporal',config=asdict(Config(epochs=args.epochs)),seeds=args.seeds,
        fingerprints=fingerprints,variants=VARIANTS,training_frames=int(selected.sum()),validation_frames=int(valid.sum()),
        validation_used_for_selection=False,selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 16')
    save_json(out/'experiment.json',plan);pooled={key(*v):np.full(len(codes),'X',dtype='<U1') for v in VARIANTS};fold_scores=[]
    for index,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop]
        remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(base,remainder,held,60*base.fps);chosen&=~near(valid,60*base.fps)
        cv_codes=np.where(chosen,remainder,3);relative=Path('cv')/f'fold_{index}';folder=out/relative
        _,_,durations=fit_group(folder,source/relative,base,audio,graph,cv_codes,chosen,[0],args.epochs,args.device)
        scores=group_scores(folder,source/relative,boosted/relative)
        half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);scored={}
        for arm,cost in VARIANTS:
            labels=decoded(*scores[arm],base,cost,durations,half);name=key(arm,cost)
            write_labels(folder/(name+'.csv'),labels);pooled[name][first:stop]=labels[first:stop]
            scored[name]=metrics(held,labels,0)
        fold_scores.append(scored);print('REFLECTED fold',index,'complete',flush=True)
    oof={name:metrics(codes,labels,0) for name,labels in pooled.items()}
    winner=max(oof,key=lambda name:oof[name]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True)
    save_json(out/'decision.json',decision)
    for name,labels in pooled.items():write_labels(out/('oof_'+name+'.csv'),labels)
    _,_,durations=fit_group(out/'global',source/'global',base,audio,graph,codes,selected,args.seeds,args.epochs,args.device)
    scores=group_scores(out/'global',source/'global',boosted/'global');arm,cost=winner.split('__')
    prediction=decoded(*scores[arm],base,float(cost),durations)
    write_labels(out/'selected_predictions.csv',prediction)
    info=json.loads((out/'global/model.json').read_text());info.update(selected=winner,input_sha256=fingerprints)
    for name in ('graph','blend'):
        np.savez_compressed(out/'global'/(name+'_scores.npz'),probability=scores[name][0],boundary=scores[name][1])
    info['control_score_sha256']={name:sha(out/'global'/(name+'_scores.npz')) for name in ('graph','blend')}
    save_json(out/'global/model.json',info)
    report=dict(experiment=plan,decision=decision,fold_scores=fold_scores,selected=winner,
                selected_main=metrics(validation,prediction,args.bootstrap))
    save_json(out/'report.json',report)
    for name,result in oof.items():print('REFLECTED OOF',name,result['spell_f1_mean'],result['n_pred_spells'],flush=True)
    print('REFLECTED MAIN',winner,report['selected_main']['spell_f1_mean'],report['selected_main']['macro_f1'],flush=True)


def predict(args):
    folder=Path(args.model);info=json.loads((folder/'model.json').read_text())
    for name in ('features','audio','graph'):
        if sha(getattr(args,name))!=info['input_sha256'][name]:raise ValueError('Reflection inference inputs changed')
    base,audio,graph=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph)
    arm,cost=info['selected'].split('__')
    if arm=='reflected':p,q=ensemble(folder,base,audio,graph,args.device)
    else:
        if sha(folder/(arm+'_scores.npz'))!=info['control_score_sha256'][arm]:raise ValueError('Frozen control scores changed')
        with np.load(folder/(arm+'_scores.npz'),allow_pickle=False) as d:p,q=d['probability'],d['boundary']
    write_labels(args.out,decoded(p,q,base,float(cost),info['durations']))
