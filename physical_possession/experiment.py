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
from boosted_possession.experiment import decoded
from .model import features,fit,probability,mix

VARIANTS=(('graph',8.),('blend',2.))+tuple((str(w),c) for w in (.5,.75,1.) for c in (1.,2.,4.,8.))
def key(arm,cost):return arm+'__'+str(int(cost))


def run(args):
    graph,trees,out=Path(args.graph_study),Path(args.boosted_study),Path(args.out)
    if out.exists():raise ValueError('Use a fresh physical study directory')
    out.mkdir(parents=True)
    base,rich=FeatureCache.load(args.features),FeatureCache.load(args.rich_features)
    if len(base.x)!=len(rich.x) or base.halves!=rich.halves:raise ValueError('Physical timeline mismatch')
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    original=json.loads((graph/'experiment.json').read_text())
    for name,field in (('train_labels','train_sha256'),('validation_labels','validation_sha256')):
        if sha(getattr(args,name))!=original[field]:raise ValueError('Physical reference files changed')
    if sha(args.features)!=original['fingerprints']['feature_sha256']:raise ValueError('Physical original features changed')
    selected,valid=training_mask(base,train,validation,60*base.fps);codes=np.where(selected,train,3)
    x,names=features(rich)
    plan=dict(kind='physical_logistic_owner_mixture',variants=VARIANTS,features=names,C=.1,intercept=False,step=5,
        fingerprints={n:sha(getattr(args,n)) for n in ('features','rich_features','train_labels','validation_labels')},
        validation_used_for_selection=False,selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 15')
    save_json(out/'experiment.json',plan);inputs={}
    def scores(relative,fit_codes):
        folder=out/relative;folder.mkdir(parents=True,exist_ok=True)
        model=fit(x,fit_codes,base.halves,threads=args.threads);owner=probability(model,x)
        np.savez_compressed(folder/'owner_model.npz',coef=model['coef'],scale=model['scale'],names=np.array(names))
        save_json(folder/'fit.json',{k:v for k,v in model.items() if k not in ('coef','scale')}|dict(model_sha256=sha(folder/'owner_model.npz')))
        np.savez_compressed(folder/'owner_scores.npz',home=owner)
        path=graph/relative/'graph_tcn/scores.npz';inputs[str(path)]=sha(path)
        with np.load(path) as d:gp,gq=d['probability'],d['boundary']
        path=trees/relative/'trees_scores.npz';inputs[str(path)]=sha(path)
        with np.load(path) as d:tp,tq=d['probability'],d['boundary']
        duration_path=graph/relative/'graph_tcn/model.json';inputs[str(duration_path)]=sha(duration_path)
        return owner,gp,gq,tp,tq,json.loads(duration_path.read_text())['durations']
    def state(arm,owner,gp,gq,tp,tq):
        if arm=='graph':return gp,gq
        if arm=='blend':return (gp+tp)/2,(gq+tq)/2
        return mix(gp,owner,float(arm)),gq
    pooled={key(*v):np.full(len(codes),'X',dtype='<U1') for v in VARIANTS};fold_scores=[]
    for i,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop]
        remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(base,remainder,held,60*base.fps);chosen&=~near(valid,60*base.fps)
        relative=Path('cv')/f'fold_{i}';data=scores(relative,np.where(chosen,remainder,3))
        half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        for arm,cost in VARIANTS:
            name=key(arm,cost);p,q=state(arm,*data[:-1]);labels=decoded(p,q,base,cost,data[-1],half)
            write_labels(out/relative/(name+'.csv'),labels);pooled[name][first:stop]=labels[first:stop]
            result[name]=metrics(held,labels,0)
        fold_scores.append(result);print('PHYSICAL fold',i,'complete',flush=True)
    oof={name:metrics(codes,labels,0) for name,labels in pooled.items()};winner=max(oof,key=lambda n:oof[n]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True)
    save_json(out/'decision.json',decision)
    for name,labels in pooled.items():write_labels(out/('oof_'+name+'.csv'),labels)
    data=scores(Path('global'),codes);arm,cost=winner.split('__');p,q=state(arm,*data[:-1])
    labels=decoded(p,q,base,float(cost),data[-1]);write_labels(out/'selected_predictions.csv',labels)
    np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q)
    save_json(out/'inputs.json',inputs)
    report=dict(experiment=plan,decision=decision,fold_scores=fold_scores,selected=winner,
        selected_main=metrics(validation,labels,args.bootstrap),durations=data[-1],selected_scores_sha256=sha(out/'selected_scores.npz'))
    save_json(out/'report.json',report)
    for name,s in oof.items():print('PHYSICAL OOF',name,s['spell_f1_mean'],s['n_pred_spells'],flush=True)
    print('PHYSICAL MAIN',winner,report['selected_main']['spell_f1_mean'],report['selected_main']['macro_f1'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('features','rich-features','train-labels','validation-labels','graph-study','boosted-study','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--threads',type=int,default=4);parser.add_argument('--bootstrap',type=int,default=1000)
    run(parser.parse_args())
