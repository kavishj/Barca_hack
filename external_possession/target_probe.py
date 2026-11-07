"""Fold-only linear target adaptation of frozen external temporal features."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask,near
from learned_possession.__main__ import read_labels,write_labels
from relational_possession.data import GraphCache
from relational_possession.model import SpatialEncoder,Config
from temporal_possession.network import deterministic,known_runs
from temporal_possession.audio import sha
from temporal_possession.experiment import metrics,save_json
from boosted_possession.experiment import decoded
from .temporal_control import pooled_features,block_context
from .dead_control import DeadControlNet
from .dead_target import interpolate_probabilities
from .temporal_target import target_state
from .turnovers import training_prior,confidence_turnovers
from .turnover_study import decode_scores

POLICIES=('fixed','typed_gated','adapted_dead','adapted_full','adapted_only')


def representations(model,x,half_ids,mean,scale,device):
    x=np.clip((x-mean)/scale,-8,8).astype(np.float32)
    output=np.zeros((len(x),2,32),np.float32);model.eval()
    with torch.inference_mode():
        for half in np.unique(half_ids):
            ix=np.flatnonzero(half_ids==half);first,stop=int(ix[0]),int(ix[-1])+1
            for start in range(first,stop,1024):
                a=block_context(x,start,1024,30,first,stop).transpose(1,2,0)
                h=model.blocks(F.gelu(model.input(torch.from_numpy(a).to(device))))
                n=min(1024,stop-start);output[start:start+n]=h[:,:,30:30+n].permute(2,0,1).cpu().numpy()
    return output


def probe_features(h,x,point):
    if h.shape!=(len(x),2,32) or x.shape!=(len(point),2,35):raise ValueError('Wrong frozen representation schema')
    margin=point[:,0]-point[:,1]
    owner=np.column_stack([h[:,0]-h[:,1],margin])
    dead=np.column_stack([h.mean(1),np.abs(h[:,0]-h[:,1]),np.abs(margin),x[:,0,33:35]])
    return owner,dead


def fit(owner,dead,labels,selected):
    selected=np.asarray(selected,bool)&(labels<3);live=selected&(labels<2)
    if not live.any() or set(labels[live].tolist())!={0,1} or not np.any(selected&(labels==2)):
        raise ValueError('Adaptation needs both owners and dead frames in training')
    owner_scale=np.maximum(np.sqrt(np.mean(owner[live].astype(np.float64)**2,axis=0)),1e-3)
    dead_mean=dead[selected].mean(0);dead_scale=np.maximum(dead[selected].std(0),1e-3)
    with threadpool_limits(limits=4):
        a=LogisticRegression(C=.1,fit_intercept=False,max_iter=2000,tol=1e-8,random_state=0).fit(
            np.clip(owner[live]/owner_scale,-8,8),(labels[live]==0).astype(int))
        b=LogisticRegression(C=.1,fit_intercept=True,max_iter=2000,tol=1e-8,random_state=0).fit(
            np.clip((dead[selected]-dead_mean)/dead_scale,-8,8).astype(np.float64),(labels[selected]==2).astype(int))
    if max(a.n_iter_[0],b.n_iter_[0])>=2000:raise ValueError('Target linear adaptation did not converge')
    return dict(owner_scale=owner_scale,owner_coef=a.coef_[0],dead_mean=dead_mean,dead_scale=dead_scale,
        dead_coef=b.coef_[0],dead_intercept=b.intercept_[0]),dict(known_samples=int(selected.sum()),live_samples=int(live.sum()),
        dead_samples=int(np.sum(selected&(labels==2))),owner_iterations=int(a.n_iter_[0]),dead_iterations=int(b.n_iter_[0]))


def probability(owner,dead,model):
    home=expit(np.clip(owner/model['owner_scale'],-8,8)@model['owner_coef'])
    d=expit(np.clip((dead-model['dead_mean'])/model['dead_scale'],-8,8)@model['dead_coef']+model['dead_intercept'])
    return np.column_stack([(1-d)*home,(1-d)*(1-home),d])


def extract(args):
    out=Path(args.out);folder=Path(args.model)
    if out.exists():raise ValueError('Use a fresh frozen-representation directory')
    plan=json.loads((folder/'experiment.json').read_text());report=json.loads((folder/'report.json').read_text())
    if not report['gate_passed'] or report['target_labels_used'] or sha(folder/'control.pt')!=report['model_sha256']:
        raise ValueError('External representation gate/provenance failed')
    if sha(args.training_features)!=plan['training_feature_sha256']['1']:raise ValueError('Wrong representation source provenance')
    with np.load(args.training_features,allow_pickle=False) as d:meta=json.loads(str(d['metadata']))
    if meta['target_labels_used'] or meta['held_out'] or sha(args.pretrained)!=meta['encoder_sha256'] or sha(args.probe)!=meta['probe_sha256']:
        raise ValueError('Wrong frozen representation encoder/probe')
    deterministic(0);graph=GraphCache.load(args.graph)
    encoder=SpatialEncoder(15,Config()).to(args.device)
    encoder.load_state_dict(torch.load(args.pretrained,map_location=args.device,weights_only=True));encoder.eval()
    with np.load(args.probe,allow_pickle=False) as d:coef=d['coef']
    x,point=pooled_features(encoder,{k:getattr(graph,k) for k in ('nodes','roles','valid','positioned','confidence')},coef,args.device)
    model=DeadControlNet().to(args.device);model.load_state_dict(torch.load(folder/'control.pt',map_location=args.device,weights_only=True))
    with np.load(folder/'normalization.npz') as d:mean,scale=d['mean'],d['scale']
    h=representations(model,x,graph.half_ids,mean,scale,args.device);owner,dead=probe_features(h,x,point)
    metadata=dict(kind='frozen_external_temporal_representations',graph_sha256=sha(args.graph),
        encoder_sha256=sha(args.pretrained),probe_sha256=sha(args.probe),temporal_sha256=sha(folder/'control.pt'),
        normalization_sha256=sha(folder/'normalization.npz'),target_labels_used=False,source_only_neural_weights=True)
    out.mkdir(parents=True);np.savez_compressed(out/'features.npz',owner=owner,dead=dead,frames=graph.frames,
        half_ids=graph.half_ids,metadata=np.array(json.dumps(metadata)))
    save_json(out/'report.json',metadata);print('FROZEN target representations extracted',flush=True)


def blend(policy,gp,gq,old_owner,turn,available,adapted,graded):
    if policy not in POLICIES:raise ValueError('Unknown target adaptation policy')
    base,q=target_state('fixed',gp,gq,old_owner,turn,available,old_owner,graded)
    if policy in POLICIES[:2]:return base,q
    p=base.copy()
    if policy=='adapted_dead':
        d=.5*(base[:,2]+adapted[:,2]);live=base[:,:2].sum(1)
        home=np.divide(base[:,0],live,out=np.full(len(base),.5),where=live>1e-12)
        value=np.column_stack([(1-d)*home,(1-d)*(1-home),d])
    elif policy=='adapted_full':value=.5*(base+adapted)
    else:value=adapted
    p[graded]=value[graded];return p,q


def run(args):
    original,matched,direct,turnovers,out=map(Path,(args.source_study,args.matched_study,args.direct_study,args.turnover_study,args.out))
    if out.exists():raise ValueError('Use a fresh target probe study directory')
    base,graph=FeatureCache.load(args.features),GraphCache.load(args.graph);source=json.loads((original/'experiment.json').read_text())
    fingerprints={k:sha(getattr(args,k)) for k in ('features','graph','representations','owner_scores','train_labels','validation_labels')}
    for k,v in (('features',source['fingerprints']['feature_sha256']),('graph',source['fingerprints']['graph_sha256']),
        ('train_labels',source['train_sha256']),('validation_labels',source['validation_sha256'])):
        if fingerprints[k]!=v:raise ValueError('Target probe input changed')
    with np.load(args.representations,allow_pickle=False) as d:
        owner,dead=d['owner'],d['dead'];meta=json.loads(str(d['metadata']));np.testing.assert_array_equal(d['frames'],graph.frames)
    if meta['target_labels_used'] or not meta['source_only_neural_weights'] or meta['graph_sha256']!=sha(args.graph):
        raise ValueError('Frozen representation supervision mismatch')
    with np.load(args.owner_scores,allow_pickle=False) as d:old_owner,turn,available=d['home'],d['boundary'],d['available']
    gated=confidence_turnovers(graph,old_owner,base)
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    selected,valid=training_mask(base,train,validation,60*base.fps);codes=np.where(selected,train,3);coarse_codes=codes[graph.frames]
    out.mkdir(parents=True);plan=dict(kind='fold_fitted_linear_external_temporal_adaptation',policies=POLICIES,fingerprints=fingerprints,
        target_neural_weights_fitted=False,validation_used_for_selection=False,C=.1,source_metadata=meta,protocol='F1_TARGET.md experiment 27',
        protocol_sha256=sha(Path(__file__).resolve().parents[1]/'F1_TARGET.md'),selection_bias=True,validation_reused=True)
    save_json(out/'experiment.json',plan);inputs={}
    def scores(relative,global_model=False):
        folder=original if global_model else matched;path=folder/relative/'graph_tcn/scores.npz'
        inputs[('original/' if global_model else 'matched/')+str(path.relative_to(folder))]=sha(path)
        with np.load(path) as d:gp,gq=d['probability'],d['boundary']
        path=original/relative/'graph_tcn/model.json';inputs['original/'+str(path.relative_to(original))]=sha(path)
        return gp,gq,json.loads(path.read_text())['durations']
    def predictions(policy,gp,gq,adapted,durations,prior,half=None):
        p,q=blend(policy,gp,gq,old_owner,turn,available,adapted,base.graded_mask())
        if policy=='typed_gated':return decode_scores(p,gq,turn,gated,base,durations,prior,policy,half)
        return decoded(p,q,base,8.,durations,half)
    pooled={p:np.full(len(codes),'X',dtype='<U1') for p in POLICIES};fold_scores=[];fits=[]
    for i,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop];remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(base,remainder,held,60*base.fps);chosen&=~near(valid,60*base.fps)
        prior,prior_stats=training_prior(np.where(chosen,remainder,3));mask=chosen[graph.frames]
        model,stats=fit(owner,dead,coarse_codes,mask)
        relative=Path('cv')/f'fold_{i}';folder=out/relative;folder.mkdir(parents=True)
        np.savez(folder/'probe.npz',**model);np.savez_compressed(folder/'training_mask.npz',selected=mask)
        adapted=interpolate_probabilities(graph,probability(owner,dead,model),base)
        np.savez_compressed(folder/'adapted_scores.npz',probability=adapted)
        fits.append(dict(**stats,prior=prior,prior_counts=prior_stats,probe_sha256=sha(folder/'probe.npz')))
        gp,gq,durations=scores(relative);half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        for policy in POLICIES:
            labels=predictions(policy,gp,gq,adapted,durations,prior,half);write_labels(folder/(policy+'.csv'),labels)
            pooled[policy][first:stop]=labels[first:stop];result[policy]=metrics(held,labels,0)
            if policy in POLICIES[:2]:
                previous=direct/relative/'1.0__8.csv' if policy=='fixed' else turnovers/relative/'typed_gated.csv'
                inputs[('direct/' if policy=='fixed' else 'turnovers/')+str(previous.relative_to(direct if policy=='fixed' else turnovers))]=sha(previous)
                old=read_labels(previous,len(codes));current=np.array([{'H':0,'A':1,'D':2,'X':3}[c] for c in labels[first:stop]])
                np.testing.assert_array_equal(old[first:stop],current)
        fold_scores.append(result);print('TARGET PROBE fold',i,'complete',flush=True)
    oof={p:metrics(codes,v,0) for p,v in pooled.items()};winner=max(oof,key=lambda p:oof[p]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True);save_json(out/'decision.json',decision)
    for p,v in pooled.items():write_labels(out/('oof_'+p+'.csv'),v)
    model,stats=fit(owner,dead,coarse_codes,selected[graph.frames]);folder=out/'global';folder.mkdir()
    np.savez(folder/'probe.npz',**model);adapted=interpolate_probabilities(graph,probability(owner,dead,model),base)
    np.savez_compressed(folder/'adapted_scores.npz',probability=adapted)
    prior,prior_stats=training_prior(codes);gp,gq,durations=scores(Path('global'),True)
    labels=predictions(winner,gp,gq,adapted,durations,prior);write_labels(out/'selected_predictions.csv',labels)
    p,q=blend(winner,gp,gq,old_owner,turn,available,adapted,base.graded_mask());np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q)
    save_json(out/'inputs.json',inputs);save_json(out/'report.json',dict(experiment=plan,decision=decision,selected=winner,fold_scores=fold_scores,
        fits=fits,global_fit=dict(**stats,prior=prior,prior_counts=prior_stats,probe_sha256=sha(folder/'probe.npz')),
        durations=durations,selected_main=metrics(validation,labels,args.bootstrap)))
    for p,s in oof.items():print('TARGET PROBE OOF',p,s['spell_f1_mean'],s['macro_f1'],flush=True)
    print('TARGET PROBE MAIN',winner,metrics(validation,labels,0)['spell_f1_mean'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('extract')
    for name in ('graph','pretrained','probe','model','training-features','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');p=sub.add_parser('compare')
    for name in ('features','graph','representations','source-study','matched-study','direct-study','turnover-study','owner-scores','train-labels','validation-labels','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--bootstrap',type=int,default=1000);args=parser.parse_args();(extract if args.command=='extract' else run)(args)
