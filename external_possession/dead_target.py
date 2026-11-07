"""Gated extraction and fixed target blends for external H/A/D probabilities."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from learned_possession.features import FeatureCache
from relational_possession.data import GraphCache
from relational_possession.model import SpatialEncoder,Config
from temporal_possession.network import deterministic
from temporal_possession.audio import sha
from temporal_possession.experiment import save_json
from .temporal_control import pooled_features,infer
from .dead_control import DeadControlNet
from .temporal_target import target_state,run

POLICIES=('fixed','typed_gated','external_0.25','external_0.5','external_1.0')


def interpolate_probabilities(graph,coarse,base):
    if coarse.shape!=(len(graph.frames),3) or not np.isfinite(coarse).all() or (coarse<0).any() or not np.allclose(coarse.sum(1),1):
        raise ValueError('Invalid external three-state probabilities')
    if graph.halves!=base.halves or graph.fps!=base.fps:raise ValueError('External three-state timeline mismatch')
    result=np.full((len(base.x),3),1/3,np.float32)
    for half,first,last in graph.halves:
        ix=graph.half_ids==half;frames=np.arange(first,last+1)
        result[first:last+1]=np.column_stack([np.interp(frames,graph.frames[ix],coarse[ix,c]) for c in range(3)])
    return result


def extract(args):
    out=Path(args.out)
    if out.exists():raise ValueError('Use a fresh external H/A/D target directory')
    folder=Path(args.model);report=json.loads((folder/'report.json').read_text());plan=json.loads((folder/'experiment.json').read_text())
    if not report['gate_passed'] or report['target_labels_used'] or sha(folder/'control.pt')!=report['model_sha256']:
        raise ValueError('External H/A/D transfer gate/provenance failed')
    if sha(args.training_features)!=plan['training_feature_sha256']['1']:raise ValueError('Wrong external H/A/D training provenance')
    with np.load(args.training_features,allow_pickle=False) as d:meta=json.loads(str(d['metadata']))
    if meta['target_labels_used'] or meta['held_out'] or sha(args.pretrained)!=meta['encoder_sha256'] or sha(args.probe)!=meta['probe_sha256']:
        raise ValueError('External H/A/D encoder/probe provenance mismatch')
    deterministic(0);base=FeatureCache.load(args.features);graph=GraphCache.load(args.graph)
    encoder=SpatialEncoder(15,Config()).to(args.device)
    encoder.load_state_dict(torch.load(args.pretrained,map_location=args.device,weights_only=True));encoder.eval()
    with np.load(args.probe,allow_pickle=False) as d:coef=d['coef']
    x,point=pooled_features(encoder,{k:getattr(graph,k) for k in ('nodes','roles','valid','positioned','confidence')},coef,args.device)
    model=DeadControlNet().to(args.device);model.load_state_dict(torch.load(folder/'control.pt',map_location=args.device,weights_only=True))
    with np.load(folder/'normalization.npz',allow_pickle=False) as d:mean,scale=d['mean'],d['scale']
    logits,_=infer(model,x,point,graph.half_ids,mean,scale,args.device)
    probability=interpolate_probabilities(graph,softmax(logits,axis=1),base)
    available=(np.nan_to_num(base.x[:,base.names.index('ball_visible')])>0)&base.graded_mask()
    metadata=dict(kind='frozen_external_three_state_temporal',target_labels_used=False,external_gate_passed=True,
        model_sha256=sha(folder/'control.pt'),normalization_sha256=sha(folder/'normalization.npz'),
        encoder_sha256=sha(args.pretrained),probe_sha256=sha(args.probe),feature_sha256=sha(args.features),graph_sha256=sha(args.graph),
        external_report_sha256=sha(folder/'report.json'),device=args.device,torch=torch.__version__)
    out.mkdir(parents=True)
    np.savez_compressed(out/'scores.npz',probability=probability,available=available,available_all=base.graded_mask(),metadata=np.array(json.dumps(metadata)))
    save_json(out/'report.json',metadata);print('EXTERNAL H/A/D target observations extracted',flush=True)


def blend_state(policy,gp,gq,old_owner,turn_q,available,external_probability,all_available):
    if policy not in POLICIES:raise ValueError('Unknown three-state blend policy')
    base,q=target_state('fixed',gp,gq,old_owner,turn_q,available,old_owner,all_available)
    if policy in POLICIES[:2]:return base,q
    external=np.asarray(external_probability)
    if external.shape!=gp.shape or not np.isfinite(external).all() or (external<0).any() or not np.allclose(external.sum(1),1):
        raise ValueError('Invalid external H/A/D target posterior')
    weight=float(policy.split('_')[1]);p=base.copy()
    p[all_available]=(1-weight)*base[all_available]+weight*external[all_available]
    return p,q


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('extract')
    for name in ('features','graph','pretrained','probe','model','training-features','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');p=sub.add_parser('compare')
    for name in ('features','graph','source-study','matched-study','direct-study','turnover-study','owner-scores','temporal-scores','train-labels','validation-labels','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--bootstrap',type=int,default=1000);args=parser.parse_args()
    if args.command=='extract':extract(args)
    else:run(args,policies=POLICIES,score_name='probability',state_fn=blend_state,
        experiment_kind='frozen_external_three_state_blends',protocol_number=26)
