"""Target-label-free direct ownership probe and observation-derived boundaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from learned_possession.features import FeatureCache
from relational_possession.data import GraphCache
from relational_possession.model import SpatialEncoder,Config
from temporal_possession.network import deterministic
from temporal_possession.audio import sha
from temporal_possession.experiment import save_json
from .data import COMMIT


def embed(encoder,data,device):
    output=[];encoder.eval()
    with torch.inference_mode():
        for first in range(0,len(data['nodes']),512):
            values=[torch.as_tensor(data[name][first:first+512],device=device) for name in ('nodes','roles','valid','positioned')]
            pooled=encoder(*values);output.append((pooled[:,0]-pooled[:,1]).cpu().numpy())
    return np.concatenate(output)


def probe_fit(difference,labels):
    if difference.shape!=(len(labels),32) or set(np.asarray(labels).tolist())!={0,1}:
        raise ValueError('External probe needs 32 embedding differences and both H/A labels')
    with threadpool_limits(limits=4):
        model=LogisticRegression(C=.1,fit_intercept=False,max_iter=2000,tol=1e-8,random_state=0).fit(
            difference.astype(np.float64),(np.asarray(labels)==0).astype(np.int8))
    if model.n_iter_[0]>=2000:raise ValueError('External probe did not converge')
    return model.coef_[0],int(model.n_iter_[0])


def fine_scores(graph,coarse_owner,cache):
    if len(coarse_owner)!=len(graph.frames) or graph.halves!=cache.halves or graph.fps!=cache.fps:
        raise ValueError('Direct probe timeline mismatch')
    available=np.nan_to_num(cache.x[:,cache.names.index('ball_visible')])>0
    owner=np.full(len(cache.x),.5,np.float32);boundary=np.zeros(len(cache.x),np.float32)
    for half,first,last in graph.halves:
        indices=np.flatnonzero(graph.half_ids==half);frames=graph.frames[indices];p=coarse_owner[indices]
        owner[first:last+1]=np.interp(np.arange(first,last+1),frames,p)
        visible=graph.positioned[indices,-1]
        for i in range(1,len(indices)):
            if not (visible[i-1] and visible[i]):continue
            point=int((frames[i-1]+frames[i])//2);jump=float(abs(p[i]-p[i-1]))
            a,b=max(first,point-3),min(last+1,point+4)
            boundary[a:b]=np.maximum(boundary[a:b],jump)
    available&=cache.graded_mask();boundary[~available]=0
    return owner,boundary,available


def extract(args):
    out=Path(args.out)
    if out.exists():raise ValueError('Use a fresh direct-probe directory')
    out.mkdir(parents=True);deterministic(0)
    info=json.loads(Path(args.pretraining_info).read_text())
    if info['target_match_labels_used'] or sha(args.pretrained)!=info['encoder_sha256']:
        raise ValueError('External encoder provenance mismatch')
    encoder=SpatialEncoder(15,Config()).to(args.device)
    encoder.load_state_dict(torch.load(args.pretrained,map_location=args.device,weights_only=True));encoder.eval()
    for p in encoder.parameters():p.requires_grad_(False)
    with np.load(args.samples,allow_pickle=False) as d:
        meta=json.loads(str(d['metadata']))
        if meta['target_match_labels_used'] or meta['source_commit']!=COMMIT:raise ValueError('Wrong external supervision source')
        data={name:d[name] for name in ('nodes','roles','valid','positioned')};labels=d['labels']
    print('DIRECT encode external',len(labels),flush=True)
    difference=embed(encoder,data,args.device);coef,iterations=probe_fit(difference,labels)
    np.savez_compressed(out/'probe.npz',coef=coef)
    base,graph=FeatureCache.load(args.features),GraphCache.load(args.graph)
    print('DIRECT encode target observations',len(graph.frames),flush=True)
    target=embed(encoder,{name:getattr(graph,name) for name in ('nodes','roles','valid','positioned')},args.device)
    owner,boundary,available=fine_scores(graph,expit(target.astype(np.float64)@coef),base)
    metadata=dict(kind='direct_external_owner_probe',source_samples_sha256=sha(args.samples),pretrained_sha256=sha(args.pretrained),
        feature_sha256=sha(args.features),graph_sha256=sha(args.graph),probe_sha256=sha(out/'probe.npz'),
        external_samples=len(labels),solver_iterations=iterations,C=.1,intercept=False,target_match_labels_used=False,
        boundary='Adjacent observed-ball owner variation; midpoint +/-3 frames; no reference labels',device=args.device,
        torch=torch.__version__)
    np.savez_compressed(out/'owner_scores.npz',home=owner,boundary=boundary,available=available,metadata=np.array(json.dumps(metadata)))
    if sha(args.pretrained)!=info['encoder_sha256']:raise ValueError('Frozen encoder file changed')
    save_json(out/'report.json',metadata)
    print('DIRECT PROBE COMPLETE',sha(out/'owner_scores.npz'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pretrained','pretraining-info','samples','features','graph','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu');extract(parser.parse_args())
