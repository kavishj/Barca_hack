"""Fixed 20-epoch graph training with immutable fold-only CLAP preprocessing."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask,near
from learned_possession.__main__ import read_labels,write_labels
from temporal_possession.audio import AudioCache,sha
from temporal_possession.network import known_runs
from temporal_possession.experiment import metrics,save_json
from external_possession.direct_study import state
from boosted_possession.experiment import decoded
from .data import GraphCache,SymmetricNormalizer
from .model import Config
from .engine import train_seed
from .experiment import fit_bundle,ensemble

POLICIES=('control','early_graph','early_external')


def source_mask(source,selected):
    with np.load(Path(source)/'training_mask.npz',allow_pickle=False) as d:
        if not np.array_equal(d['selected'],selected):raise ValueError('Short-training preprocessing mask differs from source fold')


def fit_group(folder,source,base,audio,graph,codes,selected,device):
    source,folder=Path(source),Path(folder);source_mask(source,selected)
    prior=json.loads((source/'graph_tcn/model.json').read_text())
    if prior['config']['epochs']!=60 or prior['seeds']!=[0,1,2]:raise ValueError('Use the fixed matched 60-epoch source')
    for name,digest in prior['normalizer_hashes'].items():
        if sha(source/'preprocessing'/name)!=digest:raise ValueError('Source preprocessing changed')
    configuration=dict(prior['config']);configuration['epochs']=20;config=Config(**configuration)
    folder.mkdir(parents=True);shutil.copytree(source/'preprocessing',folder/'preprocessing')
    np.savez_compressed(folder/'training_mask.npz',selected=selected)
    gx,px=SymmetricNormalizer.load(folder/'preprocessing').transform(base,audio);dest=folder/'graph_tcn'
    info=fit_bundle(dest,gx,px,graph,base,codes,selected,config,'graph_tcn',[0,1,2])
    info['durations']=prior['durations'];info['audio_sha256']=audio.provenance['cache_input_sha256']
    info['source_preprocessing_hashes']=prior['normalizer_hashes'];save_json(dest/'model.json',info)
    for seed in (0,1,2):
        print('EARLY GRAPH fit',str(folder),'seed',seed,flush=True)
        model,fit=train_seed(gx,px,graph,codes,selected,config,'graph_tcn',seed,device)
        checkpoint=dest/f'seed_{seed}.pt';torch.save(model.state_dict(),checkpoint)
        fit['checkpoint_sha256']=sha(checkpoint);save_json(dest/f'seed_{seed}.json',fit);del model
    p,q,_=ensemble(dest,base,audio,graph,device);np.savez_compressed(dest/'scores.npz',probability=p,boundary=q)
    return p,q,info['durations']


def compare(args):
    source,control,out=map(Path,(args.source_study,args.control_study,args.out))
    if out.exists():raise ValueError('Use a fresh fixed-short-training directory')
    base,audio,graph=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph)
    source_plan=json.loads((source/'experiment.json').read_text());control_report=json.loads((control/'report.json').read_text())
    control_choice=json.loads((control/'decision.json').read_text())
    if control_choice['validation_used'] or control_report['experiment']['validation_used_for_selection']:raise ValueError('Invalid short-training control selection')
    fingerprints={k:sha(getattr(args,k)) for k in ('features','audio','graph','owner_scores','train_labels','validation_labels')}
    if fingerprints!=source_plan['fingerprints']:raise ValueError('Short-training source observations or references changed')
    for k,v in control_report['experiment']['fingerprints'].items():
        if fingerprints[k]!=v:raise ValueError('Short-training control observations or references changed')
    audio.provenance['cache_input_sha256']=sha(args.audio)
    with np.load(args.owner_scores,allow_pickle=False) as d:owner,turn,available=d['home'],d['boundary'],d['available']
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    selected,valid=training_mask(base,train,validation,1500);codes=np.where(selected,train,3)
    out.mkdir(parents=True);plan=dict(kind='fixed_shorter_clap_graph_training',policies=POLICIES,epochs=20,seeds=[0,1,2],
        source_epochs=60,source_preprocessing_refitted=False,fingerprints=fingerprints,control_policy=control_choice['selected'],
        validation_used_for_selection=False,selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 30',
        protocol_sha256=sha(Path(__file__).resolve().parents[1]/'F1_TARGET.md'))
    save_json(out/'experiment.json',plan);inputs={'control/decision.json':sha(control/'decision.json'),'source/experiment.json':sha(source/'experiment.json')}
    def prediction(policy,p,q,durations,half=None):
        if policy=='early_external':p,q=state('1.0',(p,q),None,owner,turn,available)
        elif policy=='early_graph':q=np.maximum(q,turn)
        else:raise ValueError('Control predictions must come from the immutable selected control')
        return decoded(p,q,base,8.,durations,half),p,q
    pooled={p:np.full(len(codes),'X',dtype='<U1') for p in POLICIES};fold_scores=[]
    for i,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop];remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(base,remainder,held,1500);chosen&=~near(valid,1500)
        relative=Path('cv')/f'fold_{i}';folder=out/relative
        p,q,durations=fit_group(folder,source/relative,base,audio,graph,np.where(chosen,remainder,3),chosen,args.device)
        half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        previous=control/relative/(control_choice['selected']+'.csv');inputs['control/'+str(previous.relative_to(control))]=sha(previous)
        old=read_labels(previous,len(codes));labels=np.array(['H','A','D','X'])[old]
        write_labels(folder/'control.csv',labels);result['control']=metrics(held,labels,0);pooled['control'][first:stop]=labels[first:stop]
        for policy in POLICIES[1:]:
            labels,_,_=prediction(policy,p,q,durations,half);write_labels(folder/(policy+'.csv'),labels)
            pooled[policy][first:stop]=labels[first:stop];result[policy]=metrics(held,labels,0)
        fold_scores.append(result);print('EARLY GRAPH fold',i,'complete',flush=True)
    oof={p:metrics(codes,v,0) for p,v in pooled.items()};winner=max(oof,key=lambda p:oof[p]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True);save_json(out/'decision.json',decision)
    for p,v in pooled.items():write_labels(out/('oof_'+p+'.csv'),v)
    p,q,durations=fit_group(out/'global',source/'global',base,audio,graph,codes,selected,args.device)
    if winner=='control':
        shutil.copy2(control/'selected_predictions.csv',out/'selected_predictions.csv')
        shutil.copy2(control/'selected_scores.npz',out/'selected_scores.npz');main=control_report['selected_main']
        inputs['control/selected_predictions.csv']=sha(control/'selected_predictions.csv');inputs['control/selected_scores.npz']=sha(control/'selected_scores.npz')
    else:
        labels,p,q=prediction(winner,p,q,durations);write_labels(out/'selected_predictions.csv',labels)
        np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q);main=metrics(validation,labels,args.bootstrap)
    save_json(out/'inputs.json',inputs);save_json(out/'report.json',dict(experiment=plan,decision=decision,selected=winner,fold_scores=fold_scores,
        durations=durations,selected_main=main))
    for p,s in oof.items():print('EARLY GRAPH OOF',p,s['spell_f1_mean'],s['macro_f1'],flush=True)
    print('EARLY GRAPH MAIN',winner,main['spell_f1_mean'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','audio','graph','owner-scores','source-study','control-study','train-labels','validation-labels','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--bootstrap',type=int,default=1000);compare(parser.parse_args())
