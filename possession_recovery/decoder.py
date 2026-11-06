"""Positive boundary-conditioned switching, with no minimum spell duration."""
import numpy as np
from learned_possession.segment import DecoderParams,candidates


def decode(probability,boundary,fps,training_prior,switch_cost=8,grid_frames=25):
    probability=np.asarray(probability,dtype=float);boundary=np.asarray(boundary,dtype=float)
    if (probability.ndim!=2 or probability.shape[1]!=3 or not len(probability)
        or boundary.shape!=(len(probability),) or not 0<training_prior<1 or fps<=0
        or not np.isfinite(probability).all() or not np.isfinite(boundary).all()
        or (probability<0).any() or not np.allclose(probability.sum(axis=1),1)
        or ((boundary<0)|(boundary>1)).any() or switch_cost<0):
        raise ValueError('Invalid boundary-conditioned decoder inputs')
    params=DecoderParams(grid_frames=grid_frames,peak_threshold=float(np.clip(2*training_prior,.01,.35)))
    edge=candidates(probability,boundary,params)
    prefix=np.vstack([np.zeros(3),np.cumsum(np.log(np.clip(probability,1e-6,1)),axis=0)])/fps
    cost=switch_cost*training_prior/(boundary+training_prior)
    count=len(edge);best=np.full((count,3),-np.inf)
    previous=np.full((count,3),-1,int);states=np.full((count,3),-1,np.int8)
    for j in range(1,count):
        starts=edge[:j]
        for owner in range(3):
            others=np.array([x for x in range(3) if x!=owner])
            values=best[:j,others];source=np.argmax(values,axis=1)
            score=values[np.arange(j),source]-cost[starts]
            score[0]=0
            score+=prefix[edge[j],owner]-prefix[starts,owner]
            i=int(np.argmax(score));best[j,owner]=score[i]
            previous[j,owner]=i;states[j,owner]=others[source[i]] if i else -1
    owner=int(np.argmax(best[-1]));path=np.empty(len(probability),np.int8);j=count-1
    while j:
        i=previous[j,owner];path[edge[i]:edge[j]]=owner
        owner,j=int(states[j,owner]),i
    return path
