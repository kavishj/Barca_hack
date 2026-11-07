"""Matched three-seed graph training with a fixed frozen CLAP audio replacement."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from learned_possession.features import FeatureCache
from learned_possession.model import training_mask,near
from learned_possession.__main__ import read_labels,write_labels
from relational_possession.data import GraphCache,SymmetricNormalizer
from relational_possession.model import Config
from relational_possession.engine import train_seed
from relational_possession.experiment import fit_bundle,ensemble
from boosted_possession.experiment import decoded
from external_possession.direct_study import state
from external_possession.turnovers import training_prior,confidence_turnovers
from external_possession.turnover_study import decode_scores
from .audio import AudioCache,sha
from .network import known_runs
from .experiment import metrics,save_json
from .clap import MODEL,REVISION

POLICIES=('fixed','typed_gated','clap_graph','clap_external')


def fit_group(folder,prior_folder,base,audio,graph,codes,selected,config,device):
    folder=Path(folder);folder.mkdir(parents=True)
    normalizer=SymmetricNormalizer.fit(base,audio,selected);normalizer.save(folder/'preprocessing')
    gx,px=normalizer.transform(base,audio);dest=folder/'graph_tcn'
    info=fit_bundle(dest,gx,px,graph,base,codes,selected,config,'graph_tcn',[0,1,2])
    info['durations']=json.loads((Path(prior_folder)/'graph_tcn/model.json').read_text())['durations']
    info['audio_sha256']=audio.provenance['cache_input_sha256']
    save_json(dest/'model.json',info);np.savez_compressed(folder/'training_mask.npz',selected=selected)
    for seed in (0,1,2):
        print('CLAP GRAPH fit',str(folder),'seed',seed,flush=True)
        model,fit=train_seed(gx,px,graph,codes,selected,config,'graph_tcn',seed,device)
        checkpoint=dest/f'seed_{seed}.pt';torch.save(model.state_dict(),checkpoint)
        fit['checkpoint_sha256']=sha(checkpoint);save_json(dest/f'seed_{seed}.json',fit);del model
    p,q,_=ensemble(dest,base,audio,graph,device)
    np.savez_compressed(dest/'scores.npz',probability=p,boundary=q)
    return p,q,info['durations']


def compare(args):
    original,matched,direct,turnovers,out=map(Path,(args.source_study,args.matched_study,args.direct_study,args.turnover_study,args.out))
    if out.exists():raise ValueError('Use a fresh contrastive audio study directory')
    base,audio,graph=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph)
    source=json.loads((original/'experiment.json').read_text());config=Config()
    if json.loads(json.dumps(vars(config)))!=source['config']:raise ValueError('Original graph architecture changed')
    fingerprints={k:sha(getattr(args,k)) for k in ('features','audio','graph','owner_scores','train_labels','validation_labels')}
    for k,v in (('features',source['fingerprints']['feature_sha256']),('graph',source['fingerprints']['graph_sha256']),
        ('train_labels',source['train_sha256']),('validation_labels',source['validation_sha256'])):
        if fingerprints[k]!=v:raise ValueError('Contrastive audio target input changed')
    if audio.provenance['model']!=MODEL or audio.provenance['revision']!=REVISION or audio.provenance['target_labels_used'] or audio.provenance['audio_finetuned']:
        raise ValueError('Wrong frozen contrastive audio source')
    if audio.provenance['source_audio_sha256']!=source['fingerprints']['audio_sha256']:raise ValueError('Audio timeline reference changed')
    audio.provenance['cache_input_sha256']=sha(args.audio)
    with np.load(args.owner_scores,allow_pickle=False) as d:owner,turn,available=d['home'],d['boundary'],d['available']
    gated=confidence_turnovers(graph,owner,base)
    train,validation=read_labels(args.train_labels,len(base.x)),read_labels(args.validation_labels,len(base.x))
    selected,valid=training_mask(base,train,validation,60*base.fps);codes=np.where(selected,train,3)
    out.mkdir(parents=True);plan=dict(kind='matched_frozen_contrastive_audio',policies=POLICIES,seeds=[0,1,2],epochs=60,
        config=vars(config),fingerprints=fingerprints,encoder_provenance=audio.provenance,validation_used_for_selection=False,
        selection_bias=True,validation_reused=True,protocol='F1_TARGET.md experiment 28',
        protocol_sha256=sha(Path(__file__).resolve().parents[1]/'F1_TARGET.md'))
    save_json(out/'experiment.json',plan);inputs={}
    def old_scores(relative,global_model=False):
        folder=original if global_model else matched;path=folder/relative/'graph_tcn/scores.npz'
        inputs[('original/' if global_model else 'matched/')+str(path.relative_to(folder))]=sha(path)
        with np.load(path) as d:gp,gq=d['probability'],d['boundary']
        path=original/relative/'graph_tcn/model.json';inputs['original/'+str(path.relative_to(original))]=sha(path)
        return gp,gq,json.loads(path.read_text())['durations']
    def prediction(policy,old,new,durations,prior,half=None):
        gp,gq=old if policy in POLICIES[:2] else new
        if policy=='clap_graph':p,q=gp,np.maximum(gq,turn)
        else:p,q=state('1.0',(gp,gq),None,owner,turn,available)
        if policy=='typed_gated':return decode_scores(p,gq,turn,gated,base,durations,prior,policy,half),p,q
        return decoded(p,q,base,8.,durations,half),p,q
    pooled={p:np.full(len(codes),'X',dtype='<U1') for p in POLICIES};fold_scores=[];priors=[]
    for i,(first,stop) in enumerate(known_runs(codes!=3)):
        held=np.full(len(codes),3,np.int8);held[first:stop]=codes[first:stop];remainder=codes.copy();remainder[first:stop]=3
        chosen,_=training_mask(base,remainder,held,60*base.fps);chosen&=~near(valid,60*base.fps)
        cv_codes=np.where(chosen,remainder,3);prior,stats=training_prior(cv_codes);priors.append(dict(prior=prior,**stats))
        relative=Path('cv')/f'fold_{i}';folder=out/relative
        p,q,durations=fit_group(folder,original/relative,base,audio,graph,cv_codes,chosen,config,args.device)
        gp,gq,old_durations=old_scores(relative);assert durations==old_durations
        half=next(h for h,a,b in base.halves if a<=first<stop<=b+1);result={}
        for policy in POLICIES:
            labels,_,_=prediction(policy,(gp,gq),(p,q),durations,prior,half)
            write_labels(folder/(policy+'.csv'),labels);pooled[policy][first:stop]=labels[first:stop];result[policy]=metrics(held,labels,0)
            if policy in POLICIES[:2]:
                path=direct/relative/'1.0__8.csv' if policy=='fixed' else turnovers/relative/'typed_gated.csv'
                inputs[('direct/' if policy=='fixed' else 'turnovers/')+str(path.relative_to(direct if policy=='fixed' else turnovers))]=sha(path)
                old=read_labels(path,len(codes));current=np.array([{'H':0,'A':1,'D':2,'X':3}[c] for c in labels[first:stop]])
                np.testing.assert_array_equal(old[first:stop],current)
        fold_scores.append(result);print('CLAP GRAPH fold',i,'complete',flush=True)
    oof={p:metrics(codes,v,0) for p,v in pooled.items()};winner=max(oof,key=lambda p:oof[p]['spell_f1_mean'])
    decision=dict(selected=winner,oof_scores=oof,validation_used=False,selection_bias=True);save_json(out/'decision.json',decision)
    for p,v in pooled.items():write_labels(out/('oof_'+p+'.csv'),v)
    p,q,durations=fit_group(out/'global',original/'global',base,audio,graph,codes,selected,config,args.device)
    gp,gq,_=old_scores(Path('global'),True);prior,stats=training_prior(codes)
    labels,p,q=prediction(winner,(gp,gq),(p,q),durations,prior)
    write_labels(out/'selected_predictions.csv',labels);np.savez_compressed(out/'selected_scores.npz',probability=p,boundary=q)
    save_json(out/'inputs.json',inputs);save_json(out/'report.json',dict(experiment=plan,decision=decision,selected=winner,fold_scores=fold_scores,
        training_priors=priors,global_prior=dict(prior=prior,**stats),durations=durations,selected_main=metrics(validation,labels,args.bootstrap)))
    for p,s in oof.items():print('CLAP GRAPH OOF',p,s['spell_f1_mean'],s['macro_f1'],flush=True)
    print('CLAP GRAPH MAIN',winner,metrics(validation,labels,0)['spell_f1_mean'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','audio','graph','owner-scores','source-study','matched-study','direct-study','turnover-study','train-labels','validation-labels','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--bootstrap',type=int,default=1000);compare(parser.parse_args())
