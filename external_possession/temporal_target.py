"""Gated, frozen temporal-owner extraction and predeclared target ablation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask,near
from learned_possession.__main__ import read_labels,write_labels
from relational_possession.data import GraphCache
from relational_possession.model import SpatialEncoder,Config
from temporal_possession.network import deterministic,known_runs
from temporal_possession.audio import sha
from temporal_possession.experiment import metrics,save_json
from boosted_possession.experiment import decoded
from .direct import fine_scores
from .direct_study import state
from .temporal_control import ControlNet,pooled_features,infer
from .turnovers import training_prior,confidence_turnovers
from .turnover_study import decode_scores

POLICIES=('fixed','typed_gated','temporal_observed','temporal_all')


def extract(args):
    out=Path(args.out)
    if out.exists():raise ValueError('Use a fresh temporal target feature directory')
    folder=Path(args.model);report=json.loads((folder/'report.json').read_text());plan=json.loads((folder/'experiment.json').read_text())
    if not report['gate_passed'] or report['target_labels_used'] or sha(folder/'control.pt')!=report['model_sha256']:
        raise ValueError('External temporal transfer gate/provenance failed')
    if sha(args.training_features)!=plan['training_feature_sha256']['1']:raise ValueError('Wrong external training provenance')
    with np.load(args.training_features,allow_pickle=False) as d:meta=json.loads(str(d['metadata']))
    if meta['target_labels_used'] or meta['held_out'] or sha(args.pretrained)!=meta['encoder_sha256'] or sha(args.probe)!=meta['probe_sha256']:
        raise ValueError('External encoder/probe provenance mismatch')
    deterministic(0);base=FeatureCache.load(args.features);graph=GraphCache.load(args.graph)
    encoder=SpatialEncoder(15,Config()).to(args.device)
    encoder.load_state_dict(torch.load(args.pretrained,map_location=args.device,weights_only=True));encoder.eval()
    with np.load(args.probe,allow_pickle=False) as d:coef=d['coef']
    x,point=pooled_features(encoder,{k:getattr(graph,k) for k in ('nodes','roles','valid','positioned','confidence')},coef,args.device)
    model=ControlNet().to(args.device);model.load_state_dict(torch.load(folder/'control.pt',map_location=args.device,weights_only=True))
    with np.load(folder/'normalization.npz',allow_pickle=False) as d:mean,scale=d['mean'],d['scale']
    logits,q=infer(model,x,point,graph.half_ids,mean,scale,args.device)
    owner,_,available=fine_scores(graph,expit(logits[:,0]-logits[:,1]),base)
    boundary=np.zeros(len(base.x),np.float32);coarse=expit(q-np.log(plan['boundary_pos_weight']))
    for half,first,last in graph.halves:
        ix=graph.half_ids==half;boundary[first:last+1]=np.interp(np.arange(first,last+1),graph.frames[ix],coarse[ix])
    metadata=dict(kind='frozen_external_temporal_owner',target_labels_used=False,external_gate_passed=True,
        model_sha256=sha(folder/'control.pt'),normalization_sha256=sha(folder/'normalization.npz'),
        encoder_sha256=sha(args.pretrained),probe_sha256=sha(args.probe),feature_sha256=sha(args.features),graph_sha256=sha(args.graph),
        external_report_sha256=sha(folder/'report.json'),device=args.device,torch=torch.__version__)
    out.mkdir(parents=True)
    np.savez_compressed(out/'scores.npz',home=owner,available=available,available_all=base.graded_mask(),
        auxiliary_boundary=boundary,metadata=np.array(json.dumps(metadata)))
    save_json(out/'report.json',metadata);print('TEMPORAL TARGET frozen observations extracted',flush=True)


def target_state(policy,gp,gq,old_owner,turn_q,available,temporal_owner,all_available):
    if policy not in POLICIES:raise ValueError('Unknown temporal owner policy')
    owner=old_owner if policy in POLICIES[:2] else temporal_owner
    use=all_available if policy=='temporal_all' else available
    return state('1.0',(gp,gq),None,owner,turn_q,use)[0],np.maximum(gq,turn_q)


def run(args,policies=POLICIES,score_name='home',state_fn=target_state,
        experiment_kind='frozen_external_temporal_owner_ablation',protocol_number=25):
    original,matched,direct,turnovers,out=map(Path,(args.source_study,args.matched_study,args.direct_study,args.turnover_study,args.out))
    if out.exists():raise ValueError('Use a fresh temporal target comparison directory')
    base,graph=FeatureCache.load(args.features),GraphCache.load(args.graph)
    source=json.loads((original/'experiment.json').read_text())
    for name,field in (('train_labels','train_sha256'),('validation_labels','validation_sha256')):
        if sha(getattr(args,name))!=source[field]:raise ValueError('Temporal target reference changed')
    for name,key in (('features','feature_sha256'),('graph','graph_sha256')):
        if sha(getattr(args,name))!=source['fingerprints'][key]:raise ValueError('Target observation changed')
    with np.load(args.owner_scores,allow_pickle=False) as d:old_owner,turn_q,available=d['home'],d['boundary'],d['available']
    with np.load(args.temporal_scores,allow_pickle=False) as d:
        temporal_owner,all_available=d[score_name],d['available_all'];meta=json.loads(str(d['metadata']))
        np.testing.assert_array_equal(available,d['available'])
    if meta['target_labels_used'] or not meta['external_gate_passed'] or meta['feature_sha256']!=sha(args.features) or meta['graph_sha256']!=sha(args.graph):
        raise ValueError('Temporal expert gate/source mismatch')
    np.testing.assert_array_equal(all_available,base.graded_mask())
    gated=confidence_turnovers(graph,old_owner,base)
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    selected,valid=training_mask(base,train,validation,60*base.fps);codes=np.where(selected,train,3)
    out.mkdir(parents=True)
    plan=dict(kind=experiment_kind,policies=policies,feature_sha256=sha(args.features),graph_sha256=sha(args.graph),
        temporal_sha256=sha(args.temporal_scores),owner_sha256=sha(args.owner_scores),train_sha256=sha(args.train_labels),
        validation_sha256=sha(args.validation_labels),external_metadata=meta,target_refit=False,validation_used_for_selection=False,
        selection_bias=True,validation_reused=True,protocol=f'F1_TARGET.md experiment {protocol_number}',protocol_sha256=sha(Path(__file__).resolve().parents[1]/'F1_TARGET.md'))
    save_json(out/'experiment.json',plan);inputs={}
    def scores(relative,global_model=False):
        folder=original if global_model else matched;path=folder/relative/'graph_tcn/scores.npz'
        inputs[('original/' if global_model else 'matched/')+str(path.relative_to(folder))]=sha(path)
        with np.load(path) as d:p,q=d['probability'],d['boundary']
        path=original/relative/'graph_tcn/model.json';inputs['original/'+str(path.relative_to(original))]=sha(path)
        return p,q,json.loads(path.read_text())['durations']
    def prediction(policy,gp,gq,durations,prior,half=None):
        p,q=state_fn(policy,gp,gq,old_owner,turn_q,available,temporal_owner,all_available)
        if policy=='typed_gated':return decode_scores(p,gq,turn_q,gated,base,durations,prior,policy,half)
        return decoded(p,q,base,8.,durations,half)
    pooled={n:np.full(len(codes),'X',dtype='<U1') for n in policies};fold_scores=[];priors=[]
    for i,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop];remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(base,remainder,held,60*base.fps);chosen&=~near(valid,60*base.fps)
        prior,stats=training_prior(np.where(chosen,remainder,3));priors.append(dict(prior=prior,**stats))
        relative=Path('cv')/f'fold_{i}';folder=out/relative;folder.mkdir(parents=True);gp,gq,durations=scores(relative)
        half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        for policy in policies:
            labels=prediction(policy,gp,gq,durations,prior,half);write_labels(folder/(policy+'.csv'),labels)
            pooled[policy][first:stop]=labels[first:stop];result[policy]=metrics(held,labels,0)
            if policy in POLICIES[:2]:
                old=(direct/relative/'1.0__8.csv') if policy=='fixed' else (turnovers/relative/'typed_gated.csv')
                inputs[('direct/' if policy=='fixed' else 'turnovers/')+str(old.relative_to(direct if policy=='fixed' else turnovers))]=sha(old)
                previous=read_labels(old,len(codes));current=np.array([{'H':0,'A':1,'D':2,'X':3}[c] for c in labels[first:stop]])
                np.testing.assert_array_equal(previous[first:stop],current)
        fold_scores.append(result);print('TEMPORAL TARGET fold',i,'complete',flush=True)
    oof={n:metrics(codes,p,0) for n,p in pooled.items()};winner=max(oof,key=lambda n:oof[n]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True);save_json(out/'decision.json',decision)
    for n,p in pooled.items():write_labels(out/('oof_'+n+'.csv'),p)
    prior,stats=training_prior(codes);gp,gq,durations=scores(Path('global'),True)
    labels=prediction(winner,gp,gq,durations,prior);write_labels(out/'selected_predictions.csv',labels)
    p,q=state_fn(winner,gp,gq,old_owner,turn_q,available,temporal_owner,all_available)
    np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q)
    save_json(out/'inputs.json',inputs)
    save_json(out/'report.json',dict(experiment=plan,decision=decision,selected=winner,fold_scores=fold_scores,training_priors=priors,
        global_prior=dict(prior=prior,**stats),durations=durations,selected_main=metrics(validation,labels,args.bootstrap)))
    for n,s in oof.items():print('TEMPORAL TARGET OOF',n,s['spell_f1_mean'],flush=True)
    print('TEMPORAL TARGET MAIN',winner,metrics(validation,labels,0)['spell_f1_mean'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('extract')
    for name in ('features','graph','pretrained','probe','model','training-features','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');p=sub.add_parser('compare')
    for name in ('features','graph','source-study','matched-study','direct-study','turnover-study','owner-scores','temporal-scores','train-labels','validation-labels','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--bootstrap',type=int,default=1000);args=parser.parse_args();(extract if args.command=='extract' else run)(args)
