"""Fixed dead-only and full-state fusion of matched PANNs/CLAP predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask,near
from learned_possession.__main__ import read_labels,write_labels
from relational_possession.data import GraphCache
from boosted_possession.experiment import decoded
from external_possession.direct_study import state
from external_possession.turnovers import training_prior,confidence_turnovers
from external_possession.turnover_study import decode_scores
from .audio import sha
from .network import known_runs
from .experiment import metrics,save_json

POLICIES=('fixed','typed_gated','clap_external','dead_clap','dead_mean','state_mean')


def blend(policy,old,new,owner,turn,available):
    if policy not in POLICIES:raise ValueError('Unknown fixed audio fusion policy')
    gp,gq=old;new_p,nq=new
    base,bq=state('1.0',old,None,owner,turn,available)
    clap,cq=state('1.0',new,None,owner,turn,available)
    if policy in POLICIES[:2]:return base,bq
    if policy=='clap_external':return clap,cq
    q=np.maximum((gq+nq)/2,turn)
    if policy=='state_mean':return (base+clap)/2,q
    dead=new_p[:,2] if policy=='dead_clap' else .5*(gp[:,2]+new_p[:,2])
    live=base[:,:2].sum(1)
    home=np.divide(base[:,0],live,out=np.full(len(base),.5),where=live>1e-12)
    p=np.column_stack([(1-dead)*home,(1-dead)*(1-home),dead]).astype(np.float32)
    return p,cq if policy=='dead_clap' else q


def compare(args):
    original,matched,direct,turnovers,clap,out=map(Path,(args.source_study,args.matched_study,args.direct_study,args.turnover_study,args.clap_study,args.out))
    if out.exists():raise ValueError('Use a fresh fixed audio fusion directory')
    base,graph=FeatureCache.load(args.features),GraphCache.load(args.graph)
    old_plan=json.loads((original/'experiment.json').read_text());new_plan=json.loads((clap/'experiment.json').read_text())
    for name,old_key in (('train_labels','train_sha256'),('validation_labels','validation_sha256')):
        if sha(getattr(args,name))!=old_plan[old_key] or sha(getattr(args,name))!=new_plan['fingerprints'][name]:
            raise ValueError('Audio-fusion references changed')
    for name,key in (('features','feature_sha256'),('graph','graph_sha256')):
        if sha(getattr(args,name))!=old_plan['fingerprints'][key] or sha(getattr(args,name))!=new_plan['fingerprints'][name]:
            raise ValueError('Audio-fusion observation changed')
    if new_plan['seeds']!=[0,1,2] or new_plan['validation_used_for_selection']:raise ValueError('Wrong matched audio source protocol')
    if sha(args.owner_scores)!=new_plan['fingerprints']['owner_scores']:raise ValueError('External owner changed')
    with np.load(args.owner_scores,allow_pickle=False) as d:owner,turn,available=d['home'],d['boundary'],d['available']
    gated=confidence_turnovers(graph,owner,base)
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    selected,valid=training_mask(base,train,validation,1500);codes=np.where(selected,train,3)
    out.mkdir(parents=True);plan=dict(kind='fixed_panns_clap_probability_fusion',policies=POLICIES,
        fingerprints={k:sha(getattr(args,k)) for k in ('features','graph','owner_scores','train_labels','validation_labels')},
        source_weights_refitted=False,validation_used_for_selection=False,selection_bias=True,validation_reused=True,
        protocol='F1_TARGET.md experiment 29',protocol_sha256=sha(Path(__file__).resolve().parents[1]/'F1_TARGET.md'))
    save_json(out/'experiment.json',plan);inputs={'clap/experiment.json':sha(clap/'experiment.json')}
    def group(relative,global_model=False):
        folder=original if global_model else matched;path=folder/relative/'graph_tcn/scores.npz'
        inputs[('original/' if global_model else 'matched/')+str(path.relative_to(folder))]=sha(path)
        with np.load(path) as d:old=(d['probability'],d['boundary'])
        path=clap/relative/'graph_tcn/scores.npz';inputs['clap/'+str(path.relative_to(clap))]=sha(path)
        with np.load(path) as d:new=(d['probability'],d['boundary'])
        path=original/relative/'graph_tcn/model.json';inputs['original/'+str(path.relative_to(original))]=sha(path)
        durations=json.loads(path.read_text())['durations']
        assert json.loads((clap/relative/'graph_tcn/model.json').read_text())['durations']==durations
        return old,new,durations
    def prediction(policy,old,new,durations,prior,half=None):
        p,q=blend(policy,old,new,owner,turn,available)
        if policy=='typed_gated':return decode_scores(p,old[1],turn,gated,base,durations,prior,policy,half),p,q
        return decoded(p,q,base,8.,durations,half),p,q
    pooled={p:np.full(len(codes),'X',dtype='<U1') for p in POLICIES};fold_scores=[];priors=[]
    for i,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop];remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(base,remainder,held,1500);chosen&=~near(valid,1500)
        prior,stats=training_prior(np.where(chosen,remainder,3));priors.append(dict(prior=prior,**stats))
        relative=Path('cv')/f'fold_{i}';folder=out/relative;folder.mkdir(parents=True)
        old,new,durations=group(relative);half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        for policy in POLICIES:
            labels,_,_=prediction(policy,old,new,durations,prior,half);write_labels(folder/(policy+'.csv'),labels)
            pooled[policy][first:stop]=labels[first:stop];result[policy]=metrics(held,labels,0)
            if policy in POLICIES[:3]:
                source=direct if policy=='fixed' else turnovers if policy=='typed_gated' else clap
                variant='1.0__8' if policy=='fixed' else policy;path=source/relative/(variant+'.csv')
                kind='direct' if policy=='fixed' else 'turnovers' if policy=='typed_gated' else 'clap'
                inputs[kind+'/'+str(path.relative_to(source))]=sha(path)
                old_codes=read_labels(path,len(codes));current=np.array([{'H':0,'A':1,'D':2,'X':3}[c] for c in labels[first:stop]])
                np.testing.assert_array_equal(old_codes[first:stop],current)
        fold_scores.append(result);print('AUDIO FUSION fold',i,'complete',flush=True)
    oof={p:metrics(codes,v,0) for p,v in pooled.items()};winner=max(oof,key=lambda p:oof[p]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True);save_json(out/'decision.json',decision)
    for p,v in pooled.items():write_labels(out/('oof_'+p+'.csv'),v)
    old,new,durations=group(Path('global'),True);prior,stats=training_prior(codes)
    labels,p,q=prediction(winner,old,new,durations,prior);write_labels(out/'selected_predictions.csv',labels)
    np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q);save_json(out/'inputs.json',inputs)
    save_json(out/'report.json',dict(experiment=plan,decision=decision,selected=winner,fold_scores=fold_scores,training_priors=priors,
        global_prior=dict(prior=prior,**stats),durations=durations,selected_main=metrics(validation,labels,args.bootstrap)))
    for p,s in oof.items():print('AUDIO FUSION OOF',p,s['spell_f1_mean'],s['macro_f1'],flush=True)
    print('AUDIO FUSION MAIN',winner,metrics(validation,labels,0)['spell_f1_mean'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','graph','owner-scores','source-study','matched-study','direct-study','turnover-study','clap-study','train-labels','validation-labels','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--bootstrap',type=int,default=1000);compare(parser.parse_args())
