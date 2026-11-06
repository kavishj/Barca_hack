"""Fixed expanded-label baseline, typed-head/feature/context ablations and CV."""
from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch

from learned_possession.features import FeatureCache
from learned_possession.model import boundary_targets,fit_durations,training_mask,near
from learned_possession.__main__ import read_labels,write_labels
from temporal_possession.audio import AudioCache,sha
from temporal_possession.network import NetworkConfig,Normalizer,TemporalNet,deterministic
from temporal_possession.network import train_seed as baseline_train,infer_seed as baseline_infer
from temporal_possession.experiment import compare as baseline_compare,decode_scores,metrics,save_json
from learned_possession.segment import DecoderParams
from .features import known_dead
from .network import FactorizedNet,train_seed,infer_seed
from .decoder import StructuredParams,decode

LABELS=np.array(['H','A','D','X'])


def decode_all(p,q,cache,durations,params=None):
    forced=known_dead(cache)
    labels=np.full(len(cache.x),'X',dtype='<U1')
    for _,first,last in cache.halves:
        labels[first:last+1]=LABELS[decode(p[first:last+1],q[first:last+1],cache.fps,
                                          params,durations,forced[first:last+1])]
    return labels


def fit_bundle(out,cache,audio,codes,selected,config,seeds,device):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    normalizer=Normalizer.fit(cache,audio,selected)
    normalizer.save(out/'normalizer.npz')
    x=normalizer.transform(cache,audio,'learned_audio')
    durations,counts=fit_durations(codes,cache.fps,cache.halves)
    info=dict(schema=1,kind='factorized_typed_events',network=asdict(config),seeds=list(seeds),
              input_channels=x.shape[1],duration_examples=counts,durations=durations,
              decoder=asdict(StructuredParams()),decoder_algorithm='typed_local_boundary_chain_v2',
              normalizer_sha256=sha(out/'normalizer.npz'),
              training_frames=int(selected.sum()),
              known_dead_source='README.md: 110808..113741 inclusive; rule-based constraint')
    save_json(out/'model.json',info)
    for seed in seeds:
        model,fit=train_seed(x,codes,selected,cache.halves,config,seed,device)
        checkpoint=out/f'seed_{seed}.pt'
        torch.save(model.state_dict(),checkpoint)
        fit['checkpoint_sha256']=sha(checkpoint)
        save_json(out/f'seed_{seed}.json',fit)
        del model
    return info


def ensemble(out,cache,audio,device):
    out=Path(out)
    info=json.loads((out/'model.json').read_text())
    if info['schema']!=1 or info['kind']!='factorized_typed_events':
        raise ValueError('Unknown structured model schema')
    if sha(out/'normalizer.npz')!=info['normalizer_sha256']:
        raise ValueError('Structured normalizer checksum mismatch')
    normalizer=Normalizer.load(out/'normalizer.npz')
    x=normalizer.transform(cache,audio,'learned_audio')
    p,q=np.zeros((len(x),3),np.float64),np.zeros((len(x),3),np.float64)
    config=NetworkConfig(**info['network'])
    for seed in info['seeds']:
        deterministic(seed)
        fit=json.loads((out/f'seed_{seed}.json').read_text())
        path=out/f'seed_{seed}.pt'
        if sha(path)!=fit['checkpoint_sha256']:
            raise ValueError('Structured checkpoint checksum mismatch')
        model=FactorizedNet(x.shape[1],config).to(device)
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        a,b=infer_seed(model,x,cache.halves,fit['event_positive_weights'],device)
        p+=a
        q+=b
        del model
    return (p/len(info['seeds'])).astype(np.float32),(q/len(info['seeds'])).astype(np.float32),info


def cross_validation(out,base,rich,audio,train,validation,epochs,device):
    """One predeclared seed per held-out labelled window; no model selection."""
    from temporal_possession.network import known_runs
    folds=[]
    for i,(first,stop) in enumerate(known_runs(train!=3)):
        folder=out/f'fold_{i}'
        folder.mkdir(parents=True,exist_ok=True)
        held=np.full(len(train),3,np.int8)
        held[first:stop]=train[first:stop]
        remaining=train.copy()
        remaining[first:stop]=3
        selected,_=training_mask(base,remaining,held,60*base.fps)
        selected&=~near(validation!=3,60*base.fps)
        codes=np.where(selected,remaining,3)
        durations,_=fit_durations(codes,base.fps,base.halves)
        config=NetworkConfig(epochs=epochs)
        print(f'CV fold {i}: held-out [{first},{stop}), training {selected.sum()} frames',flush=True)
        normalizer=Normalizer.fit(base,audio,selected)
        x=normalizer.transform(base,audio,'learned_audio')
        by,bm=boundary_targets(codes,3)
        model,fit=baseline_train(x,codes,selected,by,bm,base.halves,config,0,device)
        torch.save(model.state_dict(),folder/'baseline_seed_0.pt')
        normalizer.save(folder/'baseline_normalizer.npz')
        save_json(folder/'baseline_fit.json',dict(fit,network=asdict(config),durations=durations))
        p,q=baseline_infer(model,x,base.halves,fit['boundary_positive_weight'],device)
        labels=decode_scores(p,q,base,durations,DecoderParams())
        labels[known_dead(base)]='D'
        write_labels(folder/'baseline.csv',labels)
        del model,x
        typed=folder/'structured'
        info=fit_bundle(typed,rich,audio,codes,selected,config,[0],device)
        p,q,_=ensemble(typed,rich,audio,device)
        write_labels(folder/'structured.csv',decode_all(p,q,rich,info['durations']))
        # Only retained for deferred scoring; held labels never fit this fold.
        folds.append((folder,held))
    return folds


def compare(args):
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    seeds=list(args.seeds)
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('Use distinct fixed ensemble seeds')
    deterministic(0)
    base,rich=FeatureCache.load(args.features),FeatureCache.load(args.rich_features)
    audio=AudioCache.load(args.audio)
    if base.halves!=rich.halves or len(base.x)!=len(rich.x):
        raise ValueError('Base and rich timelines disagree')
    train=read_labels(args.train_labels,len(base.x))
    validation=read_labels(args.validation_labels,len(base.x))
    audit=json.loads(Path(args.audit).read_text())
    if sha(args.train_labels)!=audit['train_labels_sha256'] or sha(args.validation_labels)!=audit['validation_labels_sha256']:
        raise ValueError('Expanded labels differ from the annotation audit')
    selected,valid=training_mask(base,train,validation,60*base.fps)
    codes=np.where(selected,train,3)
    configs={
        'factorized_short_base':(base,NetworkConfig(epochs=args.epochs)),
        'factorized_short_rich':(rich,NetworkConfig(epochs=args.epochs)),
        'factorized_long_rich':(rich,NetworkConfig(epochs=args.epochs,dilations=(1,2,4,8,16,32,64,128))),
    }
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    plan=dict(schema=1,kind='expanded_labels_typed_events_features_context_ablation',
              configs={name:asdict(config) for name,(_,config) in configs.items()},seeds=seeds,
              epochs_fixed=True,parameter_search=False,early_stopping=False,
              annotation_audit=audit,training_frames=int(selected.sum()),validation_frames=int(valid.sum()),
              feature_sha256=sha(args.features),rich_feature_sha256=sha(args.rich_features),audio_sha256=sha(args.audio),
              train_sha256=sha(args.train_labels),validation_sha256=sha(args.validation_labels),
              purge_seconds=60,max_observation_history_seconds=30+510/base.fps,
              evaluation_kind='exploratory_reused_draft_validation',
              source_note='No new labels; public calibration mismatch recorded but not relabelled',
              cv=bool(args.cross_validate),cv_seed=0,cv_policy='leave one existing labelled training window out',
              final_test_evaluated=False,whistle_event_reference=False,
              decoder=asdict(StructuredParams()),decoder_algorithm='typed_local_boundary_chain_v2',
              known_dead_rule='README.md supplied 110808..113741 inclusive, applied automatically')
    encoded=json.loads(json.dumps(plan))
    if (out/'experiment.json').exists() and json.loads((out/'experiment.json').read_text())!=encoded:
        raise ValueError('Output directory contains a different frozen plan')
    save_json(out/'experiment.json',plan)
    print('Structured study plan frozen before fitting or scoring',flush=True)
    started=time.monotonic()
    folds=cross_validation(out/'cv',base,rich,audio,train,validation,args.epochs,args.device) if args.cross_validate else []
    predictions={}
    for name,(cache,config) in configs.items():
        print('Training structured configuration',name,flush=True)
        arm=out/name
        info=fit_bundle(arm,cache,audio,codes,selected,config,seeds,args.device)
        info.update(feature_sha256=sha(args.features if cache is base else args.rich_features),
                    audio_sha256=sha(args.audio),input_cache='base' if cache is base else 'rich')
        save_json(arm/'model.json',info)
        p,q,_=ensemble(arm,cache,audio,args.device)
        np.savez_compressed(arm/'ensemble_scores.npz',probability=p,events=q)
        labels=decode_all(p,q,cache,info['durations'])
        write_labels(arm/'predictions.csv',labels)
        predictions[name]=labels
    # Original implementation unchanged; its first validation scoring occurs
    # after every structured/CV model above and all three baseline arms finish.
    baseline_compare(Namespace(device=args.device,epochs=args.epochs,seeds=seeds,
        features=args.features,audio=args.audio,train_labels=args.train_labels,
        validation_labels=args.validation_labels,audit=args.audit,out=str(out/'expanded_baselines'),
        bootstrap=args.bootstrap,compare=[]))
    for name in ('tracking','heuristic_audio','learned_audio'):
        path=out/'expanded_baselines'/name/'predictions.csv'
        labels=LABELS[read_labels(path,len(base.x))]
        predictions['expanded_'+name]=labels
        constrained=labels.copy()
        constrained[known_dead(base)]='D'
        write_labels(out/'expanded_baselines'/name/'constrained.csv',constrained)
        predictions['expanded_'+name+'_constrained']=constrained
    truth=np.where(base.graded_mask(),validation,3)
    scores={name:metrics(truth,labels,args.bootstrap) for name,labels in predictions.items()}
    for path in args.compare:
        scores[str(path)]=metrics(truth,LABELS[read_labels(path,len(base.x))],args.bootstrap)
    cv_scores=[]
    for folder,held in folds:
        fold=dict(fold=folder.name,known_frames=int(np.sum(held!=3)),scores={})
        for name in ('baseline','structured'):
            fold['scores'][name]=metrics(held,LABELS[read_labels(folder/f'{name}.csv',len(base.x))],0)
        cv_scores.append(fold)
    sanity={name:dict(known_dead_frames=int(known_dead(base).sum()),
                       known_dead_misses=int(np.sum(labels[known_dead(base)]!='D')))
            for name,labels in predictions.items()}
    save_json(out/'report.json',dict(experiment=plan,scores=scores,cv=cv_scores,
                                    sanity=sanity,elapsed_seconds=time.monotonic()-started))
    for name,score in scores.items():
        print(name,'spellF1',round(score['spell_f1_mean'],4),'macroF1',round(score['macro_f1'],4),
              'D_F1',round(score['per_class']['D']['f1'],4),flush=True)
    print('STRUCTURED STUDY COMPLETE',flush=True)


def predict(args):
    cache,audio=FeatureCache.load(args.features),AudioCache.load(args.audio)
    info=json.loads((Path(args.model)/'model.json').read_text())
    if sha(args.features)!=info['feature_sha256'] or sha(args.audio)!=info['audio_sha256']:
        raise ValueError('Prediction inputs differ from the saved structured model')
    p,q,_=ensemble(args.model,cache,audio,args.device)
    write_labels(args.out,decode_all(p,q,cache,info['durations'],StructuredParams(**info['decoder'])))
    print('Structured prediction SHA-256',sha(args.out),flush=True)
