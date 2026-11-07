"""Fixed AudioSet AST replacement against the immutable selected OOF control."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask,near
from learned_possession.__main__ import read_labels,write_labels
from relational_possession.data import GraphCache
from relational_possession.model import Config
from boosted_possession.experiment import decoded
from external_possession.direct_study import state
from .audio import AudioCache,sha
from .network import known_runs
from .experiment import metrics,save_json
from .clap_study import fit_group
from .ast_audio import MODEL,REVISION

POLICIES=('control','ast_graph','ast_external')


def compare(args):
    source,control,out=map(Path,(args.source_study,args.control_study,args.out))
    if out.exists():raise ValueError('Use a fresh AST replacement study directory')
    base,audio,graph=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph)
    original=json.loads((source/'experiment.json').read_text());control_report=json.loads((control/'report.json').read_text())
    choice=json.loads((control/'decision.json').read_text());config=Config()
    if choice['validation_used'] or control_report['experiment']['validation_used_for_selection']:raise ValueError('Invalid AST control selection')
    if json.loads(json.dumps(vars(config)))!=original['config']:raise ValueError('AST graph architecture/configuration changed')
    fingerprints={k:sha(getattr(args,k)) for k in ('features','audio','graph','owner_scores','train_labels','validation_labels')}
    for name,expected in (('features',original['fingerprints']['feature_sha256']),('graph',original['fingerprints']['graph_sha256']),
        ('train_labels',original['train_sha256']),('validation_labels',original['validation_sha256'])):
        if fingerprints[name]!=expected:raise ValueError('AST input observations or references changed')
    for name,expected in control_report['experiment']['fingerprints'].items():
        if fingerprints[name]!=expected:raise ValueError('AST control observations or references changed')
    meta=audio.provenance
    if meta['model']!=MODEL or meta['revision']!=REVISION or meta['target_labels_used'] or meta['audio_finetuned'] or audio.embeddings.shape[1]!=768:
        raise ValueError('Wrong fixed AST encoder or supervision provenance')
    if meta['source_audio_sha256']!=original['fingerprints']['audio_sha256']:raise ValueError('AST reference audio clock changed')
    audio.provenance['cache_input_sha256']=sha(args.audio)
    with np.load(args.owner_scores,allow_pickle=False) as d:owner,turn,available=d['home'],d['boundary'],d['available']
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    selected,valid=training_mask(base,train,validation,1500);codes=np.where(selected,train,3)
    out.mkdir(parents=True);plan=dict(kind='matched_frozen_ast_audio',policies=POLICIES,config=vars(config),epochs=60,seeds=[0,1,2],
        fingerprints=fingerprints,encoder_provenance=meta,control_policy=choice['selected'],validation_used_for_selection=False,
        selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 32',
        protocol_sha256=sha(Path(__file__).resolve().parents[1]/'F1_TARGET.md'))
    save_json(out/'experiment.json',plan);inputs={'control/decision.json':sha(control/'decision.json'),'source/experiment.json':sha(source/'experiment.json')}
    def prediction(policy,p,q,durations,half=None):
        if policy=='ast_external':p,q=state('1.0',(p,q),None,owner,turn,available)
        elif policy=='ast_graph':q=np.maximum(q,turn)
        else:raise ValueError('AST control must use its immutable saved predictions')
        return decoded(p,q,base,8.,durations,half),p,q
    pooled={p:np.full(len(codes),'X',dtype='<U1') for p in POLICIES};fold_scores=[]
    for i,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop];remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(base,remainder,held,1500);chosen&=~near(valid,1500)
        relative=Path('cv')/f'fold_{i}';folder=out/relative
        p,q,durations=fit_group(folder,source/relative,base,audio,graph,np.where(chosen,remainder,3),chosen,config,args.device,label='AST GRAPH')
        half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        previous=control/relative/(choice['selected']+'.csv');inputs['control/'+str(previous.relative_to(control))]=sha(previous)
        labels=np.array(['H','A','D','X'])[read_labels(previous,len(codes))]
        write_labels(folder/'control.csv',labels);result['control']=metrics(held,labels,0);pooled['control'][first:stop]=labels[first:stop]
        for policy in POLICIES[1:]:
            labels,_,_=prediction(policy,p,q,durations,half);write_labels(folder/(policy+'.csv'),labels)
            pooled[policy][first:stop]=labels[first:stop];result[policy]=metrics(held,labels,0)
        fold_scores.append(result);print('AST GRAPH fold',i,'complete',flush=True)
    oof={p:metrics(codes,v,0) for p,v in pooled.items()};winner=max(oof,key=lambda p:oof[p]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True);save_json(out/'decision.json',decision)
    for p,v in pooled.items():write_labels(out/('oof_'+p+'.csv'),v)
    p,q,durations=fit_group(out/'global',source/'global',base,audio,graph,codes,selected,config,args.device,label='AST GRAPH')
    if winner=='control':
        shutil.copy2(control/'selected_predictions.csv',out/'selected_predictions.csv');shutil.copy2(control/'selected_scores.npz',out/'selected_scores.npz')
        main=control_report['selected_main'];inputs['control/selected_predictions.csv']=sha(control/'selected_predictions.csv');inputs['control/selected_scores.npz']=sha(control/'selected_scores.npz')
    else:
        labels,p,q=prediction(winner,p,q,durations);write_labels(out/'selected_predictions.csv',labels)
        np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q);main=metrics(validation,labels,args.bootstrap)
    save_json(out/'inputs.json',inputs);save_json(out/'report.json',dict(experiment=plan,decision=decision,selected=winner,
        fold_scores=fold_scores,durations=durations,selected_main=main))
    for p,s in oof.items():print('AST GRAPH OOF',p,s['spell_f1_mean'],s['macro_f1'],flush=True)
    print('AST GRAPH MAIN',winner,main['spell_f1_mean'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','audio','graph','owner-scores','source-study','control-study','train-labels','validation-labels','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--bootstrap',type=int,default=1000);compare(parser.parse_args())
