"""Predeclared equal-weight mixtures of existing training-window model scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask
from learned_possession.__main__ import read_labels,write_labels
from temporal_possession.network import known_runs
from temporal_possession.audio import sha
from temporal_possession.experiment import metrics,save_json
from .experiment import decoded

VARIANTS=(('graph',8.),('graph_tree',2.))+tuple((name,cost) for name in ('graph_tcn','three_way') for cost in (1.,2.,4.,8.))
SOURCES={'graph':('graph',),'graph_tree':('graph','trees'),'graph_tcn':('graph','temporal'),'three_way':('graph','trees','temporal')}


def average(scores,names):
    if not names or any(name not in scores for name in names):raise ValueError('Unknown/empty ensemble')
    p=np.mean([scores[n][0] for n in names],axis=0,dtype=np.float64).astype(np.float32)
    q=np.mean([scores[n][1] for n in names],axis=0,dtype=np.float64).astype(np.float32)
    if p.ndim!=2 or p.shape[1]!=3 or q.shape!=(len(p),) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Invalid ensemble shape or nonfinite scores')
    if (p<0).any() or not np.allclose(p.sum(axis=1),1) or (q<0).any() or (q>1).any():
        raise ValueError('Invalid ensemble probabilities')
    return p,q


def run(args):
    out=Path(args.out)
    if out.exists():raise ValueError('Use a fresh ensemble directory')
    out.mkdir(parents=True)
    graph,trees,temporal,calibrated=map(Path,(args.graph_study,args.boosted_study,args.temporal_study,args.calibrated_study))
    base=FeatureCache.load(args.features)
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    gp=json.loads((graph/'experiment.json').read_text());tp=json.loads((temporal/'experiment.json').read_text());bp=json.loads((trees/'experiment.json').read_text())
    for name,path in (('train_sha256',args.train_labels),('validation_sha256',args.validation_labels)):
        if sha(path)!=gp[name] or gp[name]!=tp[name]:raise ValueError('Ensemble references differ')
    for name,field in (('features','feature_sha256'),('train_labels','train_sha256'),('validation_labels','validation_sha256')):
        expected=gp['fingerprints'][field] if name=='features' else gp[field]
        if sha(getattr(args,name))!=bp['fingerprints'][name] or sha(getattr(args,name))!=expected:
            raise ValueError('Graph/tree inputs differ')
    if sha(args.features)!=tp['feature_sha256']:raise ValueError('Temporal observations differ')
    selected,valid=training_mask(base,train,validation,60*base.fps);codes=np.where(selected,train,3)
    inputs={}
    def load(path,pname='probability',qname='boundary'):
        path=Path(path);inputs[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as d:return d[pname],d[qname]
    def group(relative,index=None):
        scores=dict(graph=load(graph/relative/'graph_tcn/scores.npz'),trees=load(trees/relative/'trees_scores.npz'))
        scores['temporal']=load(temporal/'expanded_baselines/learned_audio/ensemble_scores.npz') if index is None else load(calibrated/relative/'scores.npz','baseline_p','baseline_q')
        duration_path=graph/relative/'graph_tcn/model.json';inputs[str(duration_path)]=sha(duration_path)
        durations=json.loads(duration_path.read_text())['durations']
        return scores,durations
    plan=dict(kind='equal_complementary_model_ensembles',variants=VARIANTS,sources=SOURCES,
        train_sha256=sha(args.train_labels),validation_sha256=sha(args.validation_labels),feature_sha256=sha(args.features),
        validation_used_for_selection=False,selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 13')
    save_json(out/'experiment.json',plan)
    pooled={arm+'__'+str(int(cost)):np.full(len(codes),'X',dtype='<U1') for arm,cost in VARIANTS};fold_scores=[]
    for index,(first,stop) in enumerate(known_runs(codes!=3)):
        relative=Path('cv')/f'fold_{index}';folder=out/relative;folder.mkdir(parents=True)
        scores,durations=group(relative,index)
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop]
        half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        for arm,cost in VARIANTS:
            name=arm+'__'+str(int(cost));p,q=average(scores,SOURCES[arm])
            labels=decoded(p,q,base,cost,durations,half)
            pooled[name][first:stop]=labels[first:stop];write_labels(folder/(name+'.csv'),labels)
            result[name]=metrics(held,labels,0)
            if arm in ('graph','graph_tree'):
                oldname='graph__8' if arm=='graph' else 'blend__2'
                old=read_labels(trees/relative/(oldname+'.csv'),len(codes))
                np.testing.assert_array_equal(old[first:stop],np.array([{'H':0,'A':1,'D':2,'X':3}[s] for s in labels[first:stop]],np.int8))
        fold_scores.append(result);print('ENSEMBLE fold',index,'complete',flush=True)
    oof={name:metrics(codes,labels,0) for name,labels in pooled.items()}
    winner=max(oof,key=lambda name:oof[name]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True)
    save_json(out/'decision.json',decision)
    for name,labels in pooled.items():write_labels(out/('oof_'+name+'.csv'),labels)
    scores,durations=group(Path('global'));arm,cost=winner.split('__');p,q=average(scores,SOURCES[arm])
    labels=decoded(p,q,base,float(cost),durations);write_labels(out/'selected_predictions.csv',labels)
    np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q)
    save_json(out/'inputs.json',inputs)
    report=dict(experiment=plan,decision=decision,fold_scores=fold_scores,selected=winner,
        selected_main=metrics(validation,labels,args.bootstrap),selected_scores_sha256=sha(out/'selected_scores.npz'),durations=durations)
    save_json(out/'report.json',report)
    for name,s in oof.items():print('ENSEMBLE OOF',name,s['spell_f1_mean'],s['n_pred_spells'],flush=True)
    print('ENSEMBLE MAIN',winner,report['selected_main']['spell_f1_mean'],report['selected_main']['macro_f1'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','train-labels','validation-labels','graph-study','boosted-study','temporal-study','calibrated-study','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--bootstrap',type=int,default=1000)
    run(parser.parse_args())
