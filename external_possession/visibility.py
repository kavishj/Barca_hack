"""Fold-fitted ablation of marginal ball-visibility evidence in dead odds."""
import numpy as np
from scipy.special import expit


def fit(codes,visible):
    codes=np.asarray(codes);visible=np.asarray(visible,bool)
    if codes.shape!=visible.shape or codes.ndim!=1 or not np.isin(codes,range(4)).all():raise ValueError('Invalid visibility calibration labels')
    live=codes<2;dead=codes==2
    if not live.any() or not dead.any():raise ValueError('Need known live and dead calibration frames')
    counts=np.array([[np.sum(dead&~visible),np.sum(dead&visible)],
                     [np.sum(live&~visible),np.sum(live&visible)]],np.int64)
    probabilities=(counts+.5)/(counts.sum(axis=1,keepdims=True)+1)
    ratios=probabilities[0]/probabilities[1]
    return ratios,dict(counts_dead_live_by_missing_visible=counts.tolist(),smoothing=.5)


def apply(probability,visible,ratios,gamma):
    raw=np.asarray(probability);visible=np.asarray(visible,bool);ratios=np.asarray(ratios,float)
    if raw.ndim!=2 or raw.shape!=(len(visible),3) or not np.isfinite(raw).all() or (raw<0).any() or not np.allclose(raw.sum(axis=1),1):
        raise ValueError('Invalid state posterior for visibility calibration')
    if ratios.shape!=(2,) or not np.isfinite(ratios).all() or (ratios<=0).any() or not 0<=gamma<=1:raise ValueError('Invalid visibility likelihood ratios/gamma')
    if gamma==0:return raw.copy()
    p=raw.astype(np.float64);live=p[:,0]+p[:,1]
    home=np.divide(p[:,0],live,out=np.full(len(p),.5),where=live>1e-12)
    odds=np.log(np.clip(p[:,2],1e-12,1))-np.log(np.clip(live,1e-12,1))
    dead=expit(odds-gamma*np.log(ratios[visible.astype(np.int8)]))
    return np.column_stack([(1-dead)*home,(1-dead)*(1-home),dead]).astype(np.float32)
