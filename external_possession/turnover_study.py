"""Training-only comparison of four fixed turnover/stoppage policies."""
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
from structured_possession.features import known_dead
from boosted_possession.experiment import decoded
from .direct_study import state
from .turnovers import POLICIES,training_prior,confidence_turnovers,decode

LABELS=np.array(['H','A','D','X'])


def decode_scores(p,graph_q,turn_q,gated_q,cache,durations,prior,policy,half=None):
    if policy=='fixed':return decoded(p,np.maximum(graph_q,turn_q),cache,8,durations,half)
    if policy not in POLICIES:raise ValueError('Unknown turnover policy')
    out=np.full(len(cache.x),'X',dtype='<U1')
    for h,first,last in cache.halves:
        if half is not None and h!=half:continue
        effective=gated_q if policy=='typed_gated' else turn_q
        out[first:last+1]=LABELS[decode(p[first:last+1],graph_q[first:last+1],effective[first:last+1],cache.fps,
            prior,policy,durations,candidate_q=np.maximum(graph_q,turn_q)[first:last+1])]
    out[known_dead(cache)&(out!='X')]='D'
    return out


def run(args):
    original,matched,direct,out=map(Path,(args.source_study,args.matched_study,args.direct_study,args.out))
    if out.exists():raise ValueError('Use a fresh typed-turnover study directory')
    out.mkdir(parents=True)
    base,graph=FeatureCache.load(args.features),GraphCache.load(args.graph)
    source=json.loads((original/'experiment.json').read_text());control=json.loads((direct/'decision.json').read_text())
    if control['validation_used'] or control['selected']!='1.0__8':raise ValueError('Use the frozen experiment-20 selection')
    references={name:sha(getattr(args,name)) for name in ('train_labels','validation_labels')}
    for name,field in (('train_labels','train_sha256'),('validation_labels','validation_sha256')):
        if references[name]!=source[field]:raise ValueError('Typed-turnover references changed')
    if sha(args.features)!=source['fingerprints']['feature_sha256'] or sha(args.graph)!=source['fingerprints']['graph_sha256']:
        raise ValueError('Typed-turnover observations changed')
    with np.load(args.owner_scores,allow_pickle=False) as d:
        owner,turn_q,available=d['home'],d['boundary'],d['available'];metadata=json.loads(str(d['metadata']))
    if metadata['target_match_labels_used'] or metadata['graph_sha256']!=sha(args.graph) or metadata['feature_sha256']!=sha(args.features):
        raise ValueError('Wrong frozen external ownership source')
    gated_q=confidence_turnovers(graph,owner,base)
    np.savez_compressed(out/'gated_turnovers.npz',boundary=gated_q)
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    selected,valid=training_mask(base,train,validation,60*base.fps);codes=np.where(selected,train,3)
    plan=dict(kind='typed_adaptive_external_turnovers',policies=POLICIES,physical_weight=1.,switch_cost=8.,
        reference_sha256=references,feature_sha256=sha(args.features),graph_sha256=sha(args.graph),owner_sha256=sha(args.owner_scores),
        states_retrained=False,validation_used_for_selection=False,selection_bias=True,validation_reused=True,
        protocol='F1_TARGET.md experiment 21')
    save_json(out/'experiment.json',plan);inputs={'direct/decision.json':sha(direct/'decision.json')}
    def scores(relative,is_global=False):
        folder=original if is_global else matched;path=folder/relative/'graph_tcn/scores.npz'
        inputs[('original/' if is_global else 'matched/')+str(path.relative_to(folder))]=sha(path)
        with np.load(path) as d:gp,gq=d['probability'],d['boundary']
        p,_=state('1.0',(gp,gq),None,owner,turn_q,available)
        path=original/relative/'graph_tcn/model.json';inputs['original/'+str(path.relative_to(original))]=sha(path)
        return p,gq,json.loads(path.read_text())['durations']
    pooled={policy:np.full(len(codes),'X',dtype='<U1') for policy in POLICIES};fold_scores=[];priors=[]
    for index,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop]
        remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(base,remainder,held,60*base.fps);chosen&=~near(valid,60*base.fps)
        prior,stats=training_prior(np.where(chosen,remainder,3));priors.append(dict(prior=prior,**stats))
        relative=Path('cv')/f'fold_{index}';folder=out/relative;folder.mkdir(parents=True)
        p,gq,durations=scores(relative);half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        for policy in POLICIES:
            labels=decode_scores(p,gq,turn_q,gated_q,base,durations,prior,policy,half)
            write_labels(folder/(policy+'.csv'),labels);pooled[policy][first:stop]=labels[first:stop]
            result[policy]=metrics(held,labels,0)
            if policy=='fixed':
                path=direct/relative/'1.0__8.csv';inputs['direct/'+str(path.relative_to(direct))]=sha(path)
                old=read_labels(path,len(codes));pred=np.array([{'H':0,'A':1,'D':2,'X':3}[v] for v in labels[first:stop]])
                np.testing.assert_array_equal(old[first:stop],pred)
        fold_scores.append(result);print('TURNOVER fold',index,'complete',flush=True)
    oof={policy:metrics(codes,labels,0) for policy,labels in pooled.items()};winner=max(oof,key=lambda n:oof[n]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True)
    save_json(out/'decision.json',decision)
    for policy,labels in pooled.items():write_labels(out/('oof_'+policy+'.csv'),labels)
    prior,stats=training_prior(codes);p,gq,durations=scores(Path('global'),True)
    labels=decode_scores(p,gq,turn_q,gated_q,base,durations,prior,winner)
    write_labels(out/'selected_predictions.csv',labels);np.savez_compressed(out/'state_scores.npz',probability=p,graph_boundary=gq)
    if winner=='fixed':assert (out/'selected_predictions.csv').read_bytes()==(direct/'selected_predictions.csv').read_bytes()
    save_json(out/'inputs.json',inputs)
    report=dict(experiment=plan,decision=decision,fold_scores=fold_scores,training_priors=priors,selected=winner,
        global_prior=dict(prior=prior,**stats),durations=durations,selected_main=metrics(validation,labels,args.bootstrap))
    save_json(out/'report.json',report)
    for name,s in oof.items():print('TURNOVER OOF',name,s['spell_f1_mean'],s['n_pred_spells'],flush=True)
    print('TURNOVER MAIN',winner,report['selected_main']['spell_f1_mean'],report['selected_main']['macro_f1'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','graph','source-study','matched-study','direct-study','owner-scores','train-labels','validation-labels','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--bootstrap',type=int,default=1000);run(parser.parse_args())
