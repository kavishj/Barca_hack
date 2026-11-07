"""Fold-only nuisance calibration crossed with two frozen decoder choices."""
import argparse
import json
from pathlib import Path
import numpy as np
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask,near
from learned_possession.__main__ import read_labels,write_labels
from temporal_possession.network import known_runs
from temporal_possession.audio import sha
from temporal_possession.experiment import metrics,save_json
from relational_possession.data import GraphCache
from .direct_study import state
from .turnovers import POLICIES,training_prior,confidence_turnovers
from .turnover_study import decode_scores
from .visibility import fit,apply

def key(gamma,policy):return str(gamma)+'__'+policy


def run(args):
    original,matched,direct,turnovers,out=map(Path,(args.source_study,args.matched_study,args.direct_study,args.turnover_study,args.out))
    if out.exists():raise ValueError('Use a fresh visibility study directory')
    out.mkdir(parents=True)
    base,graph=FeatureCache.load(args.features),GraphCache.load(args.graph)
    source=json.loads((original/'experiment.json').read_text());choice=json.loads((turnovers/'decision.json').read_text())
    if choice['validation_used'] or choice['selected'] not in POLICIES:raise ValueError('Wrong training-only turnover control')
    policies=tuple(dict.fromkeys(('fixed',choice['selected'])));options=tuple((g,p) for g in (0.,.5,1.) for p in policies)
    fingerprints={n:sha(getattr(args,n)) for n in ('features','graph','owner_scores','train_labels','validation_labels')}
    for name,expected in (('features',source['fingerprints']['feature_sha256']),('graph',source['fingerprints']['graph_sha256']),
        ('train_labels',source['train_sha256']),('validation_labels',source['validation_sha256'])):
        if fingerprints[name]!=expected:raise ValueError('Visibility study observations/references changed')
    with np.load(args.owner_scores,allow_pickle=False) as d:owner,turn_q,available=d['home'],d['boundary'],d['available']
    expected=(np.nan_to_num(base.x[:,base.names.index('ball_visible')])>0)&base.graded_mask()
    np.testing.assert_array_equal(available,expected)
    gated_q=confidence_turnovers(graph,owner,base)
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    selected,valid=training_mask(base,train,validation,60*base.fps);codes=np.where(selected,train,3)
    plan=dict(kind='visibility_likelihood_dead_ablation',variants=options,fingerprints=fingerprints,
        turnover_control=choice['selected'],encoder_or_state_refitted=False,validation_used_for_selection=False,
        selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 22')
    save_json(out/'experiment.json',plan)
    inputs={'turnovers/decision.json':sha(turnovers/'decision.json')}
    def scores(relative,global_model=False):
        folder=original if global_model else matched;path=folder/relative/'graph_tcn/scores.npz'
        inputs[('original/' if global_model else 'matched/')+str(path.relative_to(folder))]=sha(path)
        with np.load(path) as d:gp,gq=d['probability'],d['boundary']
        path=original/relative/'graph_tcn/model.json';inputs['original/'+str(path.relative_to(original))]=sha(path)
        return gp,gq,json.loads(path.read_text())['durations']
    pooled={key(*v):np.full(len(codes),'X',dtype='<U1') for v in options};fold_scores=[];calibrations=[]
    for i,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop]
        remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(base,remainder,held,60*base.fps);chosen&=~near(valid,60*base.fps)
        fitting=np.where(chosen,remainder,3);ratios,stats=fit(fitting,available);prior,prior_stats=training_prior(fitting)
        calibrations.append(dict(ratios=ratios.tolist(),prior=prior,visibility_counts=stats,turnover_counts=prior_stats))
        relative=Path('cv')/f'fold_{i}';folder=out/relative;folder.mkdir(parents=True)
        gp,gq,durations=scores(relative);half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        for gamma,policy in options:
            calibrated=apply(gp,available,ratios,gamma);p,_=state('1.0',(calibrated,gq),None,owner,turn_q,available)
            labels=decode_scores(p,gq,turn_q,gated_q,base,durations,prior,policy,half);name=key(gamma,policy)
            write_labels(folder/(name+'.csv'),labels);pooled[name][first:stop]=labels[first:stop];result[name]=metrics(held,labels,0)
            if gamma==0:
                path=direct/relative/'1.0__8.csv' if policy=='fixed' else turnovers/relative/(policy+'.csv')
                inputs[('direct/' if policy=='fixed' else 'turnovers/')+str(path.relative_to(direct if policy=='fixed' else turnovers))]=sha(path)
                old=read_labels(path,len(codes));pred=np.array([{'H':0,'A':1,'D':2,'X':3}[v] for v in labels[first:stop]])
                np.testing.assert_array_equal(old[first:stop],pred)
        fold_scores.append(result);print('VISIBILITY fold',i,'complete',flush=True)
    oof={name:metrics(codes,labels,0) for name,labels in pooled.items()};winner=max(oof,key=lambda n:oof[n]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True)
    save_json(out/'decision.json',decision)
    for name,labels in pooled.items():write_labels(out/('oof_'+name+'.csv'),labels)
    ratios,stats=fit(codes,available);prior,prior_stats=training_prior(codes)
    gp,gq,durations=scores(Path('global'),True);gamma,policy=winner.split('__');calibrated=apply(gp,available,ratios,float(gamma))
    p,_=state('1.0',(calibrated,gq),None,owner,turn_q,available)
    labels=decode_scores(p,gq,turn_q,gated_q,base,durations,prior,policy)
    write_labels(out/'selected_predictions.csv',labels);np.savez_compressed(out/'state_scores.npz',probability=p,graph_boundary=gq)
    save_json(out/'inputs.json',inputs)
    report=dict(experiment=plan,decision=decision,fold_scores=fold_scores,fold_calibrations=calibrations,selected=winner,
        global_calibration=dict(ratios=ratios.tolist(),prior=prior,visibility_counts=stats,turnover_counts=prior_stats),
        durations=durations,selected_main=metrics(validation,labels,args.bootstrap))
    save_json(out/'report.json',report)
    for name,s in oof.items():print('VISIBILITY OOF',name,s['spell_f1_mean'],s['n_pred_spells'],flush=True)
    print('VISIBILITY MAIN',winner,report['selected_main']['spell_f1_mean'],report['selected_main']['macro_f1'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','graph','source-study','matched-study','direct-study','turnover-study','owner-scores','train-labels','validation-labels','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--bootstrap',type=int,default=1000);run(parser.parse_args())
