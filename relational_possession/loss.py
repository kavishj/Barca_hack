"""Censored interval targets and equal-per-possession auxiliary supervision."""
import numpy as np
import torch
from torch.nn import functional as F


def span_targets(codes,fps):
    n=len(codes)
    targets=np.zeros((n,2),np.float32)
    observed=np.zeros((n,2),bool)
    weights=np.zeros(n,np.float32)
    edges=np.r_[0,np.flatnonzero(np.diff(codes))+1,n]
    for first,stop in zip(edges[:-1],edges[1:]):
        if codes[first]>=2:
            continue
        frames=np.arange(first,stop)
        targets[first:stop,0]=(frames+.5-first)/fps
        targets[first:stop,1]=(stop-frames-.5)/fps
        observed[first:stop,0]=first>0 and codes[first-1]!=3
        observed[first:stop,1]=stop<n and codes[stop]!=3
        weights[first:stop]=1/(stop-first)
    return targets,observed,weights


def span_loss(predicted,classes,target,observed,weights):
    live=classes<2
    if not live.any():
        return predicted.sum()*0,predicted.sum()*0
    index=torch.arange(len(classes),device=classes.device)[live]
    prediction=predicted[index,classes[live]]
    target,observed,weights=target[live],observed[live],weights[live]
    w=weights[:,None]*observed
    regression=(F.smooth_l1_loss(torch.log1p(prediction),torch.log1p(target),reduction='none')*w).sum()/w.sum().clamp_min(1e-6)
    complete=observed.all(dim=1)
    if complete.any():
        p,t=prediction[complete],target[complete]
        intersection=torch.minimum(p,t).sum(dim=1)
        union=(p+t).sum(dim=1)-intersection
        iou=(intersection/union.clamp_min(1e-6)).clamp_min(1e-6)
        w=weights[complete]
        overlap=-(torch.log(iou)*w).sum()/w.sum().clamp_min(1e-6)
    else:
        overlap=prediction.sum()*0
    return regression,overlap
