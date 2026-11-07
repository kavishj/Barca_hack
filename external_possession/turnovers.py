"""Positive, transition-specific fees for external turnover evidence."""
import numpy as np
from learned_possession.model import near
from learned_possession.segment import DecoderParams,candidates

POLICIES=('fixed','typed_fixed','typed_adaptive','typed_gated')


def training_prior(codes):
    codes=np.asarray(codes);known=codes!=3;live=codes<2
    changes=np.zeros(len(codes),bool)
    changes[1:]=live[1:]&live[:-1]&(codes[1:]!=codes[:-1])
    target=near(changes,3);usable=live&~near(~known,4)
    count=int(usable.sum());positive=int(target[usable].sum())
    if not count:raise ValueError('Turnover prior needs usable live training frames')
    return (positive+.5)/(count+1),dict(usable_live_frames=count,positive_band_frames=positive,smoothing=.5)


def confidence_turnovers(graph,owner,cache):
    if graph.halves!=cache.halves or graph.fps!=cache.fps or len(owner)!=len(cache.x):
        raise ValueError('Confidence turnover timelines differ')
    q=np.zeros(len(owner),np.float32)
    for half,first,last in graph.halves:
        indices=np.flatnonzero(graph.half_ids==half);frames=graph.frames[indices];p=owner[frames]
        reliable=graph.positioned[indices,-1]&(graph.nodes[indices,-1,13]>.5)
        for i in range(1,len(indices)):
            if not (reliable[i-1] and reliable[i]):continue
            certainty=np.sqrt(float(graph.confidence[indices[i-1]])*float(graph.confidence[indices[i]]))
            score=float(abs(p[i]-p[i-1]))*certainty;point=int((frames[i-1]+frames[i])//2)
            a,b=max(first,point-3),min(last+1,point+4)
            q[a:b]=np.maximum(q[a:b],score)
    observed=(np.nan_to_num(cache.x[:,cache.names.index('ball_visible')])>0)&cache.graded_mask()
    q[~observed]=0
    return q


def fees(graph_q,turnover_q,prior,policy):
    if policy not in POLICIES[1:] or not 0<prior<1:raise ValueError('Invalid typed turnover policy/prior')
    graph_q,turnover_q=np.asarray(graph_q,float),np.asarray(turnover_q,float)
    if graph_q.ndim!=1 or turnover_q.shape!=graph_q.shape or not np.isfinite(graph_q).all() or not np.isfinite(turnover_q).all():
        raise ValueError('Invalid typed boundary shape/nonfinite score')
    if ((graph_q<0)|(graph_q>1)).any() or ((turnover_q<0)|(turnover_q>1)).any():raise ValueError('Invalid boundary score range')
    live_q=np.maximum(graph_q,turnover_q)
    live_cost=8. if policy=='typed_fixed' else 8*prior/(turnover_q+prior)
    live=live_cost-.5*np.log(np.clip(live_q,1e-4,1-1e-4))
    dead=8-.5*np.log(np.clip(graph_q,1e-4,1-1e-4))
    out=np.zeros((len(graph_q),3,3),np.float64)
    out[:,0,1]=out[:,1,0]=live
    out[:,0,2]=out[:,1,2]=out[:,2,0]=out[:,2,1]=dead
    return out


def decode(p,graph_q,turnover_q,fps,prior,policy,durations=None,candidate_q=None,grid_frames=25):
    p=np.asarray(p,float)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or fps<=0 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Invalid typed turnover state probabilities')
    cost=fees(graph_q,turnover_q,prior,policy)
    if len(cost)!=len(p):raise ValueError('Typed turnover state/boundary lengths differ')
    candidate_q=np.maximum(graph_q,turnover_q) if candidate_q is None else np.asarray(candidate_q)
    if candidate_q.shape!=(len(p),) or not np.isfinite(candidate_q).all() or ((candidate_q<0)|(candidate_q>1)).any():raise ValueError('Invalid shared candidate boundary scores')
    edge=candidates(p,candidate_q,DecoderParams(grid_frames=grid_frames))
    prefix=np.vstack([np.zeros(3),np.cumsum(np.log(np.clip(p,1e-6,1)),axis=0)])/fps
    count=len(edge);best=np.full((count,3),-np.inf);previous=np.full((count,3),-1,int);states=np.full((count,3),-1,np.int8)
    for j in range(1,count):
        starts=edge[:j];seconds=(edge[j]-starts)/fps
        for c in range(3):
            others=np.array([v for v in range(3) if v!=c])
            values=best[:j,others]-cost[starts[:,None],others[None,:],c]
            source=np.argmax(values,axis=1);score=values[np.arange(j),source];score[0]=0
            score+=prefix[edge[j],c]-prefix[starts,c]
            if durations and durations[c] is not None and j!=count-1:
                mu,sigma=durations[c]
                if not np.isfinite([mu,sigma]).all() or sigma<=0:raise ValueError('Invalid typed duration prior')
                penalty=-.5*((np.log(seconds)-mu)/sigma)**2;penalty[0]=0;score+=.25*penalty
            i=int(np.argmax(score));best[j,c]=score[i];previous[j,c]=i;states[j,c]=others[source[i]] if i else -1
    c=int(np.argmax(best[-1]));path=np.empty(len(p),np.int8);j=count-1
    while j:
        i=previous[j,c];path[edge[i]:edge[j]]=c;c,j=int(states[j,c]),i
    return path
