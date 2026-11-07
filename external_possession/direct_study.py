"""Matched-ensemble OOF selection for a direct external ownership expert."""
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
from boosted_possession.experiment import decoded
from physical_possession.model import mix


def variants(selected):
    arm,cost=selected.split('__')
    if arm not in ('graph','blend') or float(cost) not in (1.,2.,4.,8.):raise ValueError('Invalid matched control selection')
    result=[('graph',8.)]
    if (arm,float(cost)) not in result:result.append((arm,float(cost)))
    result.extend((str(weight),cost) for weight in (.5,.75,1.) for cost in (1.,2.,4.,8.))
    return tuple(result)


def state(arm,graph,trees,owner,boundary,available):
    gp,gq=graph
    if arm=='graph':return gp,gq
    if arm=='blend':return (gp+trees[0])/2,(gq+trees[1])/2
    p=gp.copy();expert=mix(gp,owner,float(arm));p[available]=expert[available]
    return p,np.maximum(gq,np.where(available,boundary,0))


def run(args):
    original,matched,trees,out=map(Path,(args.source_study,args.matched_study,args.boosted_study,args.out))
    if out.exists():raise ValueError('Use a fresh direct-expert study directory')
    out.mkdir(parents=True);base=FeatureCache.load(args.features)
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    old=json.loads((original/'experiment.json').read_text());match=json.loads((matched/'experiment.json').read_text())
    choice=json.loads((matched/'decision.json').read_text())
    if choice['validation_used'] or match['seeds']!=[0,1,2]:raise ValueError('Matched control selection used validation or wrong seed budget')
    for name,field in (('train_labels','train_sha256'),('validation_labels','validation_sha256')):
        if sha(getattr(args,name))!=old[field] or sha(getattr(args,name))!=match['fingerprints'][name]:raise ValueError('Direct reference mismatch')
    with np.load(args.owner_scores,allow_pickle=False) as d:
        owner,boundary,available=d['home'],d['boundary'],d['available'];metadata=json.loads(str(d['metadata']))
    if metadata['target_match_labels_used'] or metadata['feature_sha256']!=sha(args.features) or metadata['graph_sha256']!=old['fingerprints']['graph_sha256']:
        raise ValueError('External expert source/timeline mismatch')
    if owner.shape!=(len(base.x),) or boundary.shape!=owner.shape or available.dtype!=bool:raise ValueError('Invalid direct cache')
    selected,valid=training_mask(base,train,validation,60*base.fps);codes=np.where(selected,train,3)
    options=variants(choice['selected']);key=lambda arm,cost:arm+'__'+str(int(cost))
    plan=dict(kind='matched_direct_external_ownership',variants=options,matched_control=choice['selected'],
        feature_sha256=sha(args.features),train_sha256=sha(args.train_labels),validation_sha256=sha(args.validation_labels),
        owner_scores_sha256=sha(args.owner_scores),external_metadata=metadata,
        validation_used_for_selection=False,selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 20')
    save_json(out/'experiment.json',plan);inputs={'matched/decision.json':sha(matched/'decision.json')}
    def group(relative,is_global=False):
        folder=original if is_global else matched
        path=folder/relative/'graph_tcn/scores.npz';inputs[('original/' if is_global else 'matched/')+str(path.relative_to(folder))]=sha(path)
        with np.load(path) as d:graph=(d['probability'],d['boundary'])
        path=trees/relative/'trees_scores.npz';inputs['trees/'+str(path.relative_to(trees))]=sha(path)
        with np.load(path) as d:tree=(d['probability'],d['boundary'])
        path=original/relative/'graph_tcn/model.json';inputs['original/'+str(path.relative_to(original))]=sha(path)
        return graph,tree,json.loads(path.read_text())['durations']
    pooled={key(*v):np.full(len(codes),'X',dtype='<U1') for v in options};fold_scores=[]
    for i,(first,stop) in enumerate(known_runs(codes!=3)):
        relative=Path('cv')/f'fold_{i}';folder=out/relative;folder.mkdir(parents=True)
        graph,tree,durations=group(relative);held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop]
        half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        for arm,cost in options:
            p,q=state(arm,graph,tree,owner,boundary,available);labels=decoded(p,q,base,cost,durations,half)
            name=key(arm,cost);write_labels(folder/(name+'.csv'),labels);pooled[name][first:stop]=labels[first:stop]
            result[name]=metrics(held,labels,0)
            if arm in ('graph','blend'):
                old_labels=read_labels(matched/relative/(name+'.csv'),len(codes))
                codes_pred=np.array([{'H':0,'A':1,'D':2,'X':3}[v] for v in labels[first:stop]])
                np.testing.assert_array_equal(old_labels[first:stop],codes_pred)
        fold_scores.append(result);print('DIRECT OOF fold',i,'complete',flush=True)
    oof={name:metrics(codes,labels,0) for name,labels in pooled.items()};winner=max(oof,key=lambda n:oof[n]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True)
    save_json(out/'decision.json',decision)
    for name,labels in pooled.items():write_labels(out/('oof_'+name+'.csv'),labels)
    graph,tree,durations=group(Path('global'),True);arm,cost=winner.split('__')
    p,q=state(arm,graph,tree,owner,boundary,available);labels=decoded(p,q,base,float(cost),durations)
    write_labels(out/'selected_predictions.csv',labels);np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q)
    save_json(out/'inputs.json',inputs)
    report=dict(experiment=plan,decision=decision,fold_scores=fold_scores,selected=winner,durations=durations,
        selected_main=metrics(validation,labels,args.bootstrap),selected_scores_sha256=sha(out/'selected_scores.npz'))
    save_json(out/'report.json',report)
    for name,s in oof.items():print('DIRECT OOF',name,s['spell_f1_mean'],s['n_pred_spells'],flush=True)
    print('DIRECT MAIN',winner,report['selected_main']['spell_f1_mean'],report['selected_main']['macro_f1'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','owner-scores','source-study','matched-study','boosted-study','train-labels','validation-labels','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--bootstrap',type=int,default=1000);run(parser.parse_args())
