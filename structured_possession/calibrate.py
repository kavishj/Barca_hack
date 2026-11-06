"""Select switch regularisation from saved training-window holdouts only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from learned_possession.features import FeatureCache
from learned_possession.__main__ import read_labels,write_labels
from learned_possession.segment import DecoderParams,decode as basic_decode
from temporal_possession.audio import AudioCache,sha
from temporal_possession.network import NetworkConfig,Normalizer,TemporalNet,deterministic,known_runs
from temporal_possession.network import infer_seed as basic_infer
from temporal_possession.experiment import metrics,save_json
from .decoder import StructuredParams,decode
from .experiment import ensemble,decode_all
from .features import known_dead

LABELS=np.array(['H','A','D','X'])
COSTS=(1.,2.,4.,8.)


def copied_model(source,destination,cost,baseline=False):
    destination.mkdir(parents=True,exist_ok=True)
    for path in source.iterdir():
        if path.suffix in ('.pt','.json') or path.name=='normalizer.npz':
            shutil.copyfile(path,destination/path.name)
    info=json.loads((destination/'model.json').read_text())
    info['decoder']['switch_cost']=cost
    info['decoder_calibration']=dict(source='training-window OOF only',cost_grid=list(COSTS),
                                     selected_switch_cost=cost,neural_weights_changed=False)
    save_json(destination/'model.json',info)
    if baseline:
        shutil.copyfile(source.parent/'normalizer.npz',destination.parent/'normalizer.npz')


def run(args):
    deterministic(0)
    source,out=Path(args.source_study),Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA required for the requested runtime')
    base,rich=FeatureCache.load(args.features),FeatureCache.load(args.rich_features)
    audio=AudioCache.load(args.audio)
    train=read_labels(args.train_labels,len(base.x))
    source_plan=json.loads((source/'experiment.json').read_text())
    if sha(args.train_labels)!=source_plan['train_sha256']:
        raise ValueError('Calibration labels differ from the original training set')
    for field,path in (('feature_sha256',args.features),('rich_feature_sha256',args.rich_features),
                       ('audio_sha256',args.audio)):
        if sha(path)!=source_plan[field]:
            raise ValueError('Calibration observations differ from the saved training study')
    windows=known_runs(train!=3)
    plan=dict(kind='posthoc_train_only_decoder_regularisation',switch_costs=list(COSTS),
              selection='highest pooled OOF spellF1; ties use lowest cost',
              source_train_sha256=sha(args.train_labels),source_study=source_plan,
              validation_labels_read_before_selection=False,neural_weights_fitted=False,
              hypothesis='Reduce held-out over-segmentation; preserve timing/state likelihood otherwise',
              note='CV scores select one parameter and are not an unbiased estimate after selection')
    save_json(out/'calibration_plan.json',plan)
    pooled={name:{cost:np.full(len(train),'X',dtype='<U1') for cost in COSTS}
            for name in ('baseline','structured')}
    started=time.monotonic()
    for i,(first,stop) in enumerate(windows):
        folder=source/'cv'/f'fold_{i}'
        target=out/'cv'/f'fold_{i}'
        target.mkdir(parents=True,exist_ok=True)
        halves=[half for half in base.halves if half[1]<=first and stop<=half[2]+1]
        if len(halves)!=1:
            raise ValueError('A held-out window crosses a half boundary')
        _,left,right=halves[0]
        normalizer=Normalizer.load(folder/'baseline_normalizer.npz')
        x=normalizer.transform(base,audio,'learned_audio')
        fit=json.loads((folder/'baseline_fit.json').read_text())
        model=TemporalNet(x.shape[1],NetworkConfig(**fit['network'])).to(args.device)
        model.load_state_dict(torch.load(folder/'baseline_seed_0.pt',map_location=args.device,weights_only=True))
        # Preserve the original batching as well as the half-isolated context.
        p,q=basic_infer(model,x,base.halves,fit['boundary_positive_weight'],args.device)
        del x,model
        sp,sq,info=ensemble(folder/'structured',rich,audio,args.device)
        np.savez_compressed(target/'scores.npz',baseline_p=p,baseline_q=q,structured_p=sp,structured_q=sq)
        for cost in COSTS:
            for name,cache,prob,event,durations in (
                ('baseline',base,p,q,fit['durations']),
                ('structured',rich,sp,sq,info['durations'])):
                if name=='baseline':
                    path=basic_decode(prob[left:right+1],event[left:right+1],cache.fps,
                                      DecoderParams(switch_cost=cost),durations)
                else:
                    path=decode(prob[left:right+1],event[left:right+1],cache.fps,
                                StructuredParams(switch_cost=cost),durations,known_dead(cache)[left:right+1])
                labels=np.full(len(train),'X',dtype='<U1')
                labels[left:right+1]=LABELS[path]
                labels[known_dead(cache)]='D'
                write_labels(target/f'{name}_cost_{cost:g}.csv',labels)
                pooled[name][cost][first:stop]=labels[first:stop]
                if cost==1:
                    old=read_labels(folder/f'{name}.csv',len(train))
                    np.testing.assert_array_equal(LABELS[old[first:stop]],labels[first:stop])
        print(f'Calibrated OOF fold {i+1}/{len(windows)}; original cost-1 predictions reproduced',flush=True)
    cv={name:{cost:metrics(train,labels,0) for cost,labels in grid.items()} for name,grid in pooled.items()}
    selected={name:max(COSTS,key=lambda cost:cv[name][cost]['spell_f1_mean']) for name in cv}
    # Save the decision before reading or scoring any main validation labels.
    save_json(out/'selected_decoder.json',dict(costs=selected,selection='pooled training-window OOF spellF1',
                                               validation_used=False,cv=cv))
    print('Training-only selected costs:',selected,flush=True)
    for name,grid in pooled.items():
        for cost,labels in grid.items():
            write_labels(out/f'oof_{name}_cost_{cost:g}.csv',labels)
    predictions={}
    for name in ('baseline','structured'):
        model=source/'expanded_baselines/learned_audio' if name=='baseline' else source/'factorized_short_rich'
        info=json.loads((model/'model.json').read_text())
        with np.load(model/'ensemble_scores.npz',allow_pickle=False) as scores:
            p=scores['probability']
            q=scores['boundary' if name=='baseline' else 'events']
        if name=='baseline':
            labels=np.full(len(train),'X',dtype='<U1')
            for _,first,last in base.halves:
                labels[first:last+1]=LABELS[basic_decode(p[first:last+1],q[first:last+1],base.fps,
                    DecoderParams(**(info['decoder']|dict(switch_cost=selected[name]))),info['durations'])]
            labels[known_dead(base)]='D'
            copied_model(model,out/'models/baseline/learned_audio',selected[name],baseline=True)
        else:
            labels=decode_all(p,q,rich,info['durations'],
                              StructuredParams(**(info['decoder']|dict(switch_cost=selected[name]))))
            copied_model(model,out/'models/structured',selected[name])
        predictions[name]=labels
        write_labels(out/f'{name}.csv',labels)
    validation=read_labels(args.validation_labels,len(train))
    if sha(args.validation_labels)!=source_plan['validation_sha256']:
        raise ValueError('Main validation reference changed')
    scores={name:metrics(validation,labels,args.bootstrap) for name,labels in predictions.items()}
    save_json(out/'report.json',dict(plan=plan,selected_costs=selected,cv=cv,scores=scores,
                                    elapsed_seconds=time.monotonic()-started))
    for name,score in scores.items():
        print(name,'selected cost',selected[name],'spellF1',round(score['spell_f1_mean'],4),
              'macroF1',round(score['macro_f1'],4),flush=True)
    print('TRAIN-ONLY DECODER CALIBRATION COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-study','features','rich-features','audio','train-labels','validation-labels','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--bootstrap',type=int,default=1000)
    run(parser.parse_args())
