"""Unambiguous supplied attacking-phase priors, limited to missing-ball ownership."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask
from learned_possession.__main__ import read_labels,write_labels
from temporal_possession.audio import sha
from temporal_possession.network import known_runs
from temporal_possession.experiment import metrics,save_json
from external_possession.direct_study import state
from boosted_possession.experiment import decoded
from .data import TEAM_NAMES
from .signals import POSSESSION_CODES

POLICIES=('control','hint_mean','hint_only')


def phase_hints(path,cache):
    counts=np.zeros((len(cache.x),2),np.int16)
    for event in ET.parse(path).getroot().iter('instance'):
        if str(event.findtext('code')).strip().upper() not in POSSESSION_CODES:continue
        labels={str(p.findtext('group')).strip().casefold():str(p.findtext('text')).strip() for p in event.findall('label')}
        team=TEAM_NAMES.get(labels.get('team','').casefold(),-1)
        half={'1st half':1,'2nd half':2}.get(labels.get('half','').casefold())
        block=[(a,b) for h,a,b in cache.halves if h==half]
        if team not in (0,1) or len(block)!=1:continue
        first,last=block[0];a=max(first,round(float(event.findtext('start'))*cache.fps))
        b=min(last+1,round(float(event.findtext('end'))*cache.fps))
        if b>a:counts[a:b,team]+=1
    hint=np.full(len(cache.x),-1,np.int8)
    hint[(counts[:,0]>0)&(counts[:,1]==0)]=0
    hint[(counts[:,1]>0)&(counts[:,0]==0)]=1
    return hint,dict(unique_hint_frames=int((hint>=0).sum()),conflict_frames=int(np.all(counts>0,axis=1).sum()))


def apply(policy,probability,hint,missing):
    if policy not in POLICIES:raise ValueError('Unknown fixed phase-hint policy')
    result=probability.copy()
    if policy=='control':return result
    selected=np.asarray(missing,bool)&(hint>=0)
    mass=probability[:,:2].sum(1)
    home=np.divide(probability[:,0],mass,out=np.full(len(mass),.5),where=mass>1e-12)
    prior=np.where(hint==0,.75,.25)
    if policy=='hint_mean':prior=.5*(home+prior)
    live=1-probability[selected,2]
    result[selected,0]=live*prior[selected];result[selected,1]=live*(1-prior[selected])
    return result


def compare(args):
    source,control,out=map(Path,(args.source_study,args.control_study,args.out))
    if out.exists():raise ValueError('Use a fresh fixed phase-hint comparison directory')
    base=FeatureCache.load(args.features);source_plan=json.loads((source/'experiment.json').read_text())
    report_control=json.loads((control/'report.json').read_text());choice=json.loads((control/'decision.json').read_text())
    fingerprints={k:sha(getattr(args,k)) for k in ('features','owner_scores','tags','train_labels','validation_labels')}
    for k in ('features','owner_scores','train_labels','validation_labels'):
        if fingerprints[k]!=source_plan['fingerprints'][k] or fingerprints[k]!=report_control['experiment']['fingerprints'][k]:
            raise ValueError('Phase hint source observations or references changed')
    if choice['validation_used'] or source_plan['validation_used_for_selection']:raise ValueError('Phase hint source selection changed')
    hint,stats=phase_hints(args.tags,base)
    missing=(np.nan_to_num(base.x[:,base.names.index('ball_visible')])<=0)&base.graded_mask()
    stats['eligible_missing_frames']=int(np.sum(missing&(hint>=0)))
    with np.load(args.owner_scores,allow_pickle=False) as d:owner,turn,available=d['home'],d['boundary'],d['available']
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    selected,_=training_mask(base,train,validation,1500);codes=np.where(selected,train,3)
    out.mkdir(parents=True);plan=dict(kind='fixed_automatic_phase_missing_ball_hint',policies=POLICIES,fingerprints=fingerprints,
        hint_probability=.75,weights_or_scalers_refitted=False,reference_labels_used_for_hint=False,control_policy=choice['selected'],
        validation_used_for_selection=False,selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 33',
        protocol_sha256=sha(Path(__file__).resolve().parents[1]/'F1_TARGET.md'),hint_statistics=stats)
    save_json(out/'experiment.json',plan);np.savez_compressed(out/'hints.npz',hint=hint,missing=missing)
    inputs={'control/decision.json':sha(control/'decision.json')}
    def group(relative):
        path=source/relative/'graph_tcn/scores.npz';inputs['source/'+str(path.relative_to(source))]=sha(path)
        with np.load(path) as d:p,q=state('1.0',(d['probability'],d['boundary']),None,owner,turn,available)
        path=source/relative/'graph_tcn/model.json';inputs['source/'+str(path.relative_to(source))]=sha(path)
        return p,q,json.loads(path.read_text())['durations']
    pooled={p:np.full(len(codes),'X',dtype='<U1') for p in POLICIES};fold_scores=[]
    for i,(first,stop) in enumerate(known_runs(codes!=3)):
        relative=Path('cv')/f'fold_{i}';folder=out/relative;folder.mkdir(parents=True)
        p,q,durations=group(relative);half=next(h for h,a,b in base.halves if a<=first<stop<=b+1)
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop];result={}
        previous=control/relative/(choice['selected']+'.csv');inputs['control/'+str(previous.relative_to(control))]=sha(previous)
        previous_codes=read_labels(previous,len(codes))
        for policy in POLICIES:
            posterior=apply(policy,p,hint,missing);labels=decoded(posterior,q,base,8.,durations,half)
            write_labels(folder/(policy+'.csv'),labels);pooled[policy][first:stop]=labels[first:stop];result[policy]=metrics(held,labels,0)
            if policy=='control':
                now=np.array([{'H':0,'A':1,'D':2,'X':3}[c] for c in labels[first:stop]])
                np.testing.assert_array_equal(previous_codes[first:stop],now)
        fold_scores.append(result);print('PHASE HINT fold',i,'complete',flush=True)
    oof={p:metrics(codes,v,0) for p,v in pooled.items()};winner=max(oof,key=lambda p:oof[p]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True);save_json(out/'decision.json',decision)
    for p,v in pooled.items():write_labels(out/('oof_'+p+'.csv'),v)
    p,q,durations=group(Path('global'))
    if winner=='control':
        shutil.copy2(control/'selected_predictions.csv',out/'selected_predictions.csv');shutil.copy2(control/'selected_scores.npz',out/'selected_scores.npz')
        main=report_control['selected_main'];inputs['control/selected_predictions.csv']=sha(control/'selected_predictions.csv');inputs['control/selected_scores.npz']=sha(control/'selected_scores.npz')
    else:
        p=apply(winner,p,hint,missing);labels=decoded(p,q,base,8.,durations);write_labels(out/'selected_predictions.csv',labels)
        np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q);main=metrics(validation,labels,args.bootstrap)
    save_json(out/'inputs.json',inputs);save_json(out/'report.json',dict(experiment=plan,decision=decision,selected=winner,
        fold_scores=fold_scores,durations=durations,selected_main=main))
    for p,s in oof.items():print('PHASE HINT OOF',p,s['spell_f1_mean'],s['macro_f1'],flush=True)
    print('PHASE HINT MAIN',winner,main['spell_f1_mean'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','owner-scores','tags','source-study','control-study','train-labels','validation-labels','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--bootstrap',type=int,default=1000);compare(parser.parse_args())
