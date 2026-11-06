"""Calibrate on the same three-seed ensemble used at deployment."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask,near
from learned_possession.__main__ import read_labels,write_labels
from temporal_possession.network import known_runs
from temporal_possession.audio import AudioCache,sha
from temporal_possession.experiment import metrics,save_json
from relational_possession.data import GraphCache,SymmetricNormalizer
from relational_possession.model import Config
from relational_possession.engine import train_seed,infer_seed
from relational_possession.experiment import ensemble
from boosted_possession.experiment import decoded

VARIANTS=tuple((arm,cost) for arm in ('graph','blend') for cost in (1.,2.,4.,8.))
def key(arm,cost):return arm+'__'+str(int(cost))


def compare(args):
    source,trees,out=Path(args.source_study),Path(args.boosted_study),Path(args.out)
    if out.exists():raise ValueError('Use a fresh matched-ensemble directory')
    out.mkdir(parents=True)
    base,audio,graph=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph)
    original=json.loads((source/'experiment.json').read_text());config=Config(epochs=args.epochs)
    if json.loads(json.dumps(vars(config)))!=original['config'] or original['seeds']!=[0,1,2]:
        raise ValueError('Use original graph configuration and three deployment seeds')
    fingerprints={n:sha(getattr(args,n)) for n in ('features','audio','graph','train_labels','validation_labels')}
    expected=dict(features=original['fingerprints']['feature_sha256'],audio=original['fingerprints']['audio_sha256'],
        graph=original['fingerprints']['graph_sha256'],train_labels=original['train_sha256'],validation_labels=original['validation_sha256'])
    if fingerprints!=expected:raise ValueError('Matched-ensemble observations or references changed')
    tree_plan=json.loads((trees/'experiment.json').read_text())
    for name in ('features','audio','train_labels','validation_labels'):
        if tree_plan['fingerprints'][name]!=fingerprints[name]:raise ValueError('Matched tree control input mismatch')
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    selected,valid=training_mask(base,train,validation,60*base.fps);codes=np.where(selected,train,3)
    plan=dict(kind='matched_three_seed_oof_calibration',config=vars(config),seeds=[0,1,2],variants=VARIANTS,
        fingerprints=fingerprints,global_weights_refitted=False,validation_used_for_selection=False,
        selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 19')
    save_json(out/'experiment.json',plan);inputs={}
    pooled={key(*v):np.full(len(codes),'X',dtype='<U1') for v in VARIANTS};fold_scores=[]
    for index,(first,stop) in enumerate(known_runs(codes!=3)):
        relative=Path('cv')/f'fold_{index}';folder=out/relative;folder.mkdir(parents=True)
        saved=source/relative/'graph_tcn';info=json.loads((saved/'model.json').read_text());info['seeds']=[0,1,2]
        for path in [saved/'model.json',saved/'seed_0.pt',saved/'seed_0.json']+list((saved.parent/'preprocessing').glob('*.npz')):
            inputs['graph/'+str(path.relative_to(source))]=sha(path)
        if json.loads(json.dumps(info['config']))!=original['config']:raise ValueError('Original fold config mismatch')
        shutil.copytree(saved.parent/'preprocessing',folder/'preprocessing')
        model_folder=folder/'graph_tcn';model_folder.mkdir()
        for suffix in ('pt','json'):shutil.copy2(saved/f'seed_0.{suffix}',model_folder/f'seed_0.{suffix}')
        gx,px=SymmetricNormalizer.load(folder/'preprocessing').transform(base,audio)
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop]
        remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(base,remainder,held,60*base.fps);chosen&=~near(valid,60*base.fps)
        cv_codes=np.where(chosen,remainder,3)
        inputs['graph/'+str((saved/'scores.npz').relative_to(source))]=sha(saved/'scores.npz')
        with np.load(saved/'scores.npz') as d:p,q=d['probability'].astype(np.float64),d['boundary'].astype(np.float64)
        for seed in (1,2):
            print('MATCHED fit',index,'seed',seed,flush=True)
            model,fit=train_seed(gx,px,graph,cv_codes,chosen,config,'graph_tcn',seed,args.device)
            checkpoint=model_folder/f'seed_{seed}.pt';torch.save(model.state_dict(),checkpoint)
            fit['checkpoint_sha256']=sha(checkpoint);save_json(model_folder/f'seed_{seed}.json',fit)
            a,b=infer_seed(model,gx,px,graph,fit['boundary_positive_weight'],args.device);del model
            np.savez_compressed(model_folder/f'seed_{seed}_scores.npz',probability=a,boundary=b);p+=a;q+=b
        gp,gq=(p/3).astype(np.float32),(q/3).astype(np.float32)
        save_json(model_folder/'model.json',info);np.savez_compressed(model_folder/'scores.npz',probability=gp,boundary=gq)
        path=trees/relative/'trees_scores.npz';inputs['trees/'+str(path.relative_to(trees))]=sha(path)
        with np.load(path) as d:tp,tq=d['probability'],d['boundary']
        half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        for arm,cost in VARIANTS:
            p,q=(gp,gq) if arm=='graph' else ((gp+tp)/2,(gq+tq)/2)
            labels=decoded(p,q,base,cost,info['durations'],half);name=key(arm,cost)
            write_labels(folder/(name+'.csv'),labels);pooled[name][first:stop]=labels[first:stop];result[name]=metrics(held,labels,0)
        fold_scores.append(result);print('MATCHED fold',index,'complete',flush=True)
    oof={name:metrics(codes,labels,0) for name,labels in pooled.items()};winner=max(oof,key=lambda n:oof[n]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True)
    save_json(out/'decision.json',decision)
    for name,labels in pooled.items():write_labels(out/('oof_'+name+'.csv'),labels)
    global_folder=source/'global/graph_tcn';info=json.loads((global_folder/'model.json').read_text())
    if info['seeds']!=[0,1,2]:raise ValueError('Global model must average seeds 0/1/2')
    for path in [global_folder/'model.json']+list(global_folder.glob('seed_*'))+list((global_folder.parent/'preprocessing').glob('*.npz')):
        inputs['graph/'+str(path.relative_to(source))]=sha(path)
    path=global_folder/'scores.npz';inputs['graph/'+str(path.relative_to(source))]=sha(path)
    with np.load(path) as d:gp,gq=d['probability'],d['boundary']
    path=trees/'global/trees_scores.npz';inputs['trees/'+str(path.relative_to(trees))]=sha(path)
    with np.load(path) as d:tp,tq=d['probability'],d['boundary']
    arm,cost=winner.split('__');p,q=(gp,gq) if arm=='graph' else ((gp+tp)/2,(gq+tq)/2)
    labels=decoded(p,q,base,float(cost),info['durations']);write_labels(out/'selected_predictions.csv',labels)
    np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q);save_json(out/'inputs.json',inputs)
    report=dict(experiment=plan,decision=decision,fold_scores=fold_scores,selected=winner,durations=info['durations'],
        selected_main=metrics(validation,labels,args.bootstrap),selected_scores_sha256=sha(out/'selected_scores.npz'))
    save_json(out/'report.json',report)
    for name,s in oof.items():print('MATCHED OOF',name,s['spell_f1_mean'],s['n_pred_spells'],flush=True)
    print('MATCHED MAIN',winner,report['selected_main']['spell_f1_mean'],report['selected_main']['macro_f1'],flush=True)


def predict(args):
    folder=Path(args.model);report=json.loads((folder/'report.json').read_text());plan=report['experiment']
    for name in ('features','audio','graph'):
        if sha(getattr(args,name))!=plan['fingerprints'][name]:raise ValueError('Matched inference inputs changed')
    for name,digest in json.loads((folder/'inputs.json').read_text()).items():
        kind,relative=name.split('/',1)
        root=Path(args.source_study if kind=='graph' else args.boosted_study)
        if sha(root/relative)!=digest:raise ValueError('Matched source artifact changed')
    base,audio,graph=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph)
    p,q,info=ensemble(Path(args.source_study)/'global/graph_tcn',base,audio,graph,args.device)
    arm,cost=report['selected'].split('__')
    if arm=='blend':
        with np.load(Path(args.boosted_study)/'global/trees_scores.npz') as d:p,q=(p+d['probability'])/2,(q+d['boundary'])/2
    write_labels(args.out,decoded(p,q,base,float(cost),info['durations']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    for command in ('compare','predict'):
        p=commands.add_parser(command)
        for name in ('features','audio','graph','source-study','boosted-study','out'):p.add_argument('--'+name,required=True)
        p.add_argument('--device',default='cpu')
        if command=='compare':
            for name in ('train-labels','validation-labels'):p.add_argument('--'+name,required=True)
            p.add_argument('--epochs',type=int,default=60);p.add_argument('--bootstrap',type=int,default=1000)
        else:p.add_argument('--model',required=True)
    args=parser.parse_args();compare(args) if args.command=='compare' else predict(args)
