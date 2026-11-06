"""Fixed source-dependent turnover/stoppage costs; no new fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask
from learned_possession.segment import decode,DecoderParams
from learned_possession.__main__ import read_labels,write_labels
from temporal_possession.network import known_runs
from temporal_possession.audio import sha
from temporal_possession.experiment import metrics,save_json
from structured_possession.features import known_dead
from boosted_possession.experiment import decoded
from .model import mix

VARIANTS=(('graph',8.,8.),('blend',2.,2.))+tuple((source,live,dead) for source in ('graph','physical') for live in (1.,2.,4.) for dead in (4.,8.))
LABELS=np.array(['H','A','D','X'])
def key(arm,live,dead):return f'{arm}__{int(live)}_{int(dead)}'


def matrix(live,dead):
    if min(live,dead)<0:raise ValueError('Transition fees must be nonnegative')
    return np.array([[0,live,dead],[live,0,dead],[dead,dead,0]],float)


def typed_decoded(p,q,cache,live,dead,durations,half=None):
    labels=np.full(len(cache.x),'X',dtype='<U1')
    for h,first,last in cache.halves:
        if half is None or half==h:
            labels[first:last+1]=LABELS[decode(p[first:last+1],q[first:last+1],cache.fps,
                DecoderParams(switch_cost=0),durations,transition_costs=matrix(live,dead))]
    labels[known_dead(cache)&(labels!='X')]='D'
    return labels


def run(args):
    graph,trees,physical,out=map(Path,(args.graph_study,args.boosted_study,args.physical_study,args.out))
    if out.exists():raise ValueError('Use a fresh typed-cost directory')
    out.mkdir(parents=True);base=FeatureCache.load(args.features)
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    original=json.loads((graph/'experiment.json').read_text());expert=json.loads((physical/'experiment.json').read_text())
    for name,field in (('train_labels','train_sha256'),('validation_labels','validation_sha256')):
        if sha(getattr(args,name))!=original[field] or sha(getattr(args,name))!=expert['fingerprints'][name]:raise ValueError('Typed reference mismatch')
    if sha(args.features)!=original['fingerprints']['feature_sha256']:raise ValueError('Typed original observations changed')
    selected,valid=training_mask(base,train,validation,60*base.fps);codes=np.where(selected,train,3)
    plan=dict(kind='typed_turnover_stoppage_costs',variants=VARIANTS,physical_owner_weight=.75,
        train_sha256=sha(args.train_labels),validation_sha256=sha(args.validation_labels),feature_sha256=sha(args.features),
        validation_used_for_selection=False,selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 18')
    save_json(out/'experiment.json',plan);inputs={}
    def load(path,pname='probability',qname='boundary'):
        inputs[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as d:return d[pname],d[qname]
    def group(relative):
        gp,gq=load(graph/relative/'graph_tcn/scores.npz');tp,tq=load(trees/relative/'trees_scores.npz')
        path=physical/relative/'owner_scores.npz';inputs[str(path)]=sha(path)
        with np.load(path) as d:owner=d['home']
        duration_path=graph/relative/'graph_tcn/model.json';inputs[str(duration_path)]=sha(duration_path)
        return dict(graph=(gp,gq),blend=((gp+tp)/2,(gq+tq)/2),physical=(mix(gp,owner,.75),gq)),json.loads(duration_path.read_text())['durations']
    pooled={key(*v):np.full(len(codes),'X',dtype='<U1') for v in VARIANTS};fold_scores=[]
    def prediction(variant,scores,durations,half=None):
        arm,live,dead=variant;p,q=scores[arm]
        return decoded(p,q,base,live,durations,half) if live==dead else typed_decoded(p,q,base,live,dead,durations,half)
    for index,(first,stop) in enumerate(known_runs(codes!=3)):
        relative=Path('cv')/f'fold_{index}';folder=out/relative;folder.mkdir(parents=True)
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop]
        scores,durations=group(relative);half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        for variant in VARIANTS:
            name=key(*variant);labels=prediction(variant,scores,durations,half)
            write_labels(folder/(name+'.csv'),labels);pooled[name][first:stop]=labels[first:stop];result[name]=metrics(held,labels,0)
            if variant in VARIANTS[:2]:
                old=read_labels(trees/relative/('graph__8.csv' if variant[0]=='graph' else 'blend__2.csv'),len(codes))
                np.testing.assert_array_equal(old[first:stop],np.array([{'H':0,'A':1,'D':2,'X':3}[v] for v in labels[first:stop]]))
        fold_scores.append(result);print('TYPED fold',index,'complete',flush=True)
    oof={name:metrics(codes,labels,0) for name,labels in pooled.items()};winner=max(oof,key=lambda n:oof[n]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True)
    save_json(out/'decision.json',decision)
    for name,labels in pooled.items():write_labels(out/('oof_'+name+'.csv'),labels)
    scores,durations=group(Path('global'));variant=next(v for v in VARIANTS if key(*v)==winner)
    labels=prediction(variant,scores,durations);write_labels(out/'selected_predictions.csv',labels)
    p,q=scores[variant[0]];np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q)
    save_json(out/'inputs.json',inputs)
    report=dict(experiment=plan,decision=decision,fold_scores=fold_scores,selected=winner,selected_variant=variant,
        selected_main=metrics(validation,labels,args.bootstrap),durations=durations,selected_scores_sha256=sha(out/'selected_scores.npz'))
    save_json(out/'report.json',report)
    for name,s in oof.items():print('TYPED OOF',name,s['spell_f1_mean'],s['n_pred_spells'],flush=True)
    print('TYPED MAIN',winner,report['selected_main']['spell_f1_mean'],report['selected_main']['macro_f1'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','train-labels','validation-labels','graph-study','boosted-study','physical-study','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--bootstrap',type=int,default=1000);run(parser.parse_args())
