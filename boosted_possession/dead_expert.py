"""Fixed tabular dead-ball probabilities with an independent frozen owner."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from learned_possession.features import FeatureCache
from learned_possession.__main__ import read_labels,write_labels
from learned_possession.model import training_mask
from temporal_possession.audio import sha
from temporal_possession.network import known_runs
from temporal_possession.experiment import metrics,save_json
from .experiment import decoded

POLICIES=('control','trees_dead','trees_graph_mean','trees_audio_mean')


def factorized(policy,old,clap,trees,owner,turn,available):
    if policy not in POLICIES[1:]:raise ValueError('Unknown fixed dead-ball expert')
    if policy=='trees_dead':dead,q=trees[0][:,2],trees[1]
    elif policy=='trees_graph_mean':dead,q=.5*(trees[0][:,2]+old[0][:,2]),.5*(trees[1]+old[1])
    else:dead,q=(trees[0][:,2]+old[0][:,2]+clap[0][:,2])/3,(trees[1]+old[1]+clap[1])/3
    live=old[0][:,:2].sum(1)
    home=np.divide(old[0][:,0],live,out=np.full(len(live),.5),where=live>1e-12)
    # Preserve the independent owner even when old P(D) rounded to exactly one.
    home[available]=owner[available]
    return np.column_stack([(1-dead)*home,(1-dead)*(1-home),dead]).astype(np.float32),np.maximum(q,turn)


def compare(args):
    original,matched,clap,trees,control,out=map(Path,(args.source_study,args.matched_study,args.clap_study,args.tree_study,args.control_study,args.out))
    if out.exists():raise ValueError('Use a fresh factorized dead-ball expert directory')
    base=FeatureCache.load(args.features);source=json.loads((original/'experiment.json').read_text())
    tree_plan=json.loads((trees/'experiment.json').read_text());clap_plan=json.loads((clap/'experiment.json').read_text())
    control_report=json.loads((control/'report.json').read_text());choice=json.loads((control/'decision.json').read_text())
    fingerprints={k:sha(getattr(args,k)) for k in ('features','owner_scores','train_labels','validation_labels')}
    for k,expected in (('features',source['fingerprints']['feature_sha256']),('train_labels',source['train_sha256']),('validation_labels',source['validation_sha256'])):
        if fingerprints[k]!=expected or fingerprints[k]!=tree_plan['fingerprints'][k] or fingerprints[k]!=clap_plan['fingerprints'][k]:
            raise ValueError('Factorized expert observations or references changed')
    if fingerprints['owner_scores']!=clap_plan['fingerprints']['owner_scores'] or choice['validation_used']:
        raise ValueError('Factorized external owner/control source changed')
    with np.load(args.owner_scores,allow_pickle=False) as d:owner,turn,available=d['home'],d['boundary'],d['available']
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    selected,_=training_mask(base,train,validation,1500);codes=np.where(selected,train,3)
    out.mkdir(parents=True);plan=dict(kind='fixed_factorized_tabular_dead_expert',policies=POLICIES,fingerprints=fingerprints,
        control_policy=choice['selected'],weights_or_scalers_refitted=False,validation_used_for_selection=False,
        selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 31',
        protocol_sha256=sha(Path(__file__).resolve().parents[1]/'F1_TARGET.md'))
    save_json(out/'experiment.json',plan);inputs={'control/decision.json':sha(control/'decision.json'),'trees/experiment.json':sha(trees/'experiment.json')}
    def group(relative,global_model=False):
        folder=original if global_model else matched;path=folder/relative/'graph_tcn/scores.npz'
        inputs[('original/' if global_model else 'matched/')+str(path.relative_to(folder))]=sha(path)
        with np.load(path) as d:old=(d['probability'],d['boundary'])
        values=[]
        for name,root,filename in (('clap',clap,'graph_tcn/scores.npz'),('trees',trees,'trees_scores.npz')):
            path=root/relative/filename;inputs[name+'/'+str(path.relative_to(root))]=sha(path)
            with np.load(path) as d:values.append((d['probability'],d['boundary']))
        path=original/relative/'graph_tcn/model.json';inputs['original/'+str(path.relative_to(original))]=sha(path)
        return old,*values,json.loads(path.read_text())['durations']
    pooled={p:np.full(len(codes),'X',dtype='<U1') for p in POLICIES};fold_scores=[]
    for i,(first,stop) in enumerate(known_runs(codes!=3)):
        relative=Path('cv')/f'fold_{i}';folder=out/relative;folder.mkdir(parents=True)
        old,new,tree,durations=group(relative);half=next(h for h,a,b in base.halves if a<=first<stop<=b+1)
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop];result={}
        previous=control/relative/(choice['selected']+'.csv');inputs['control/'+str(previous.relative_to(control))]=sha(previous)
        previous_codes=read_labels(previous,len(codes));labels=np.array(['H','A','D','X'])[previous_codes]
        write_labels(folder/'control.csv',labels);pooled['control'][first:stop]=labels[first:stop];result['control']=metrics(held,labels,0)
        for policy in POLICIES[1:]:
            p,q=factorized(policy,old,new,tree,owner,turn,available);labels=decoded(p,q,base,8.,durations,half)
            write_labels(folder/(policy+'.csv'),labels);pooled[policy][first:stop]=labels[first:stop];result[policy]=metrics(held,labels,0)
        fold_scores.append(result);print('DEAD EXPERT fold',i,'complete',flush=True)
    oof={p:metrics(codes,v,0) for p,v in pooled.items()};winner=max(oof,key=lambda p:oof[p]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True);save_json(out/'decision.json',decision)
    for p,v in pooled.items():write_labels(out/('oof_'+p+'.csv'),v)
    old,new,tree,durations=group(Path('global'),True)
    if winner=='control':
        shutil.copy2(control/'selected_predictions.csv',out/'selected_predictions.csv');shutil.copy2(control/'selected_scores.npz',out/'selected_scores.npz')
        main=control_report['selected_main'];inputs['control/selected_predictions.csv']=sha(control/'selected_predictions.csv');inputs['control/selected_scores.npz']=sha(control/'selected_scores.npz')
    else:
        p,q=factorized(winner,old,new,tree,owner,turn,available);labels=decoded(p,q,base,8.,durations)
        write_labels(out/'selected_predictions.csv',labels);np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q)
        main=metrics(validation,labels,args.bootstrap)
    save_json(out/'inputs.json',inputs);save_json(out/'report.json',dict(experiment=plan,decision=decision,selected=winner,
        fold_scores=fold_scores,durations=durations,selected_main=main))
    for p,s in oof.items():print('DEAD EXPERT OOF',p,s['spell_f1_mean'],s['macro_f1'],flush=True)
    print('DEAD EXPERT MAIN',winner,main['spell_f1_mean'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','owner-scores','source-study','matched-study','clap-study','tree-study','control-study','train-labels','validation-labels','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--bootstrap',type=int,default=1000);compare(parser.parse_args())
