"""Typed segment transitions, hard metadata facts, frame-level second pass."""
from dataclasses import dataclass
import numpy as np
from learned_possession.segment import DecoderParams, candidates


@dataclass(frozen=True)
class StructuredParams(DecoderParams):
    refinement_frames:int=25


def transition_kind(previous,current):
    if previous==current:
        raise ValueError('No event for an unchanged state')
    return 1 if previous==2 else (0 if current==2 else 2)


def objective(path,probability,events,fps,p,durations=None,forced=None):
    """Independent complete-path objective, used by exhaustive decoder checks."""
    edges=np.r_[0,np.flatnonzero(np.diff(path))+1,len(path)]
    total=0.0
    for a,b in zip(edges[:-1],edges[1:]):
        c=int(path[a])
        if forced is not None and c!=2 and np.any(forced[a:b]):
            return -np.inf
        total+=np.log(np.clip(probability[a:b,c],1e-6,1)).sum()/fps
        if a:
            kind=transition_kind(int(path[a-1]),c)
            total+=p.boundary_weight*np.log(np.clip(events[a,kind],1e-4,1-1e-4))-p.switch_cost
        if a and b<len(path) and durations and durations[c] is not None:
            mu,sigma=durations[c]
            total-=0.5*p.duration_weight*((np.log((b-a)/fps)-mu)/sigma)**2
    return total


def solve(probability,events,fps,p,durations,forced,edge):
    n=len(probability)
    prefix=np.vstack([np.zeros(3),np.cumsum(np.log(np.clip(probability,1e-6,1)),axis=0)])/fps
    forbidden=np.r_[0,np.cumsum(forced)]
    q=np.log(np.clip(events,1e-4,1-1e-4))*p.boundary_weight
    best=np.full((len(edge),3),-np.inf)
    previous_edge=np.full((len(edge),3),-1,int)
    previous_state=np.full((len(edge),3),-1,int)
    for j in range(1,len(edge)):
        starts=edge[:j]
        seconds=(edge[j]-starts)/fps
        for c in range(3):
            others=np.array([k for k in range(3) if k!=c])
            typed=np.column_stack([q[starts,transition_kind(k,c)] for k in others])
            prev=best[:j,others]+typed-p.switch_cost
            src=np.argmax(prev,axis=1)
            values=prev[np.arange(j),src]
            values[0]=0
            values+=prefix[edge[j],c]-prefix[starts,c]
            if c!=2:
                values[(forbidden[edge[j]]-forbidden[starts])>0]=-np.inf
            if durations and durations[c] is not None and j!=len(edge)-1:
                mu,sigma=durations[c]
                if not np.isfinite([mu,sigma]).all() or sigma<=0:
                    raise ValueError('Invalid duration distribution')
                penalty=-0.5*((np.log(seconds)-mu)/sigma)**2
                penalty[0]=0
                values+=p.duration_weight*penalty
            i=int(np.argmax(values))
            best[j,c]=values[i]
            previous_edge[j,c]=i
            previous_state[j,c]=others[src[i]] if i else -1
    if not np.isfinite(best[-1]).any():
        raise ValueError('No feasible structured path')
    c,j=int(np.argmax(best[-1])),len(edge)-1
    path=np.empty(n,np.int8)
    while j:
        i=previous_edge[j,c]
        path[edge[i]:edge[j]]=c
        c,j=int(previous_state[j,c]),i
    return path


def refine(probability,events,fps,p,durations,forced,coarse):
    """Joint boundary optimisation with the coarse state sequence fixed.

    The chain costs O(number of boundaries * (2R+1)^2), including overlapping
    neighbourhoods. The full coarse path remains feasible. This avoids building
    one quadratic graph over every fine candidate in a long, fragmented half.
    """
    n=len(coarse)
    edges=np.r_[0,np.flatnonzero(np.diff(coarse))+1,n]
    states=coarse[edges[:-1]]
    choices=[np.array([0]),
             *(np.arange(max(1,t-p.refinement_frames),min(n,t+p.refinement_frames+1)) for t in edges[1:-1]),
             np.array([n])]
    prefix=np.vstack([np.zeros(3),np.cumsum(np.log(np.clip(probability,1e-6,1)),axis=0)])/fps
    forbidden=np.r_[0,np.cumsum(forced)]
    q=np.log(np.clip(events,1e-4,1-1e-4))*p.boundary_weight
    score=np.array([0.])
    predecessors=[]
    for i,c in enumerate(states):
        starts,stops=choices[i],choices[i+1]
        length=stops[None,:]-starts[:,None]
        values=score[:,None]+prefix[stops,c][None,:]-prefix[starts,c][:,None]
        if i:
            values+=q[starts,transition_kind(int(states[i-1]),int(c))][:,None]-p.switch_cost
        if durations and durations[c] is not None and i and i<len(states)-1:
            mu,sigma=durations[c]
            if not np.isfinite([mu,sigma]).all() or sigma<=0:
                raise ValueError('Invalid duration distribution')
            values-=0.5*p.duration_weight*((np.log(np.maximum(length,1)/fps)-mu)/sigma)**2
        valid=length>0
        if c!=2:
            valid&=(forbidden[stops][None,:]-forbidden[starts][:,None])==0
        values[~valid]=-np.inf
        previous=np.argmax(values,axis=0)
        score=values[previous,np.arange(len(stops))]
        predecessors.append(previous)
    if not np.isfinite(score[0]):
        raise ValueError('No feasible local boundary refinement')
    path=np.empty(n,np.int8)
    selected=0
    for i in range(len(states)-1,-1,-1):
        before=int(predecessors[i][selected])
        path[choices[i][before]:choices[i+1][selected]]=states[i]
        selected=before
    return path


def decode(probability,events,fps,p=None,durations=None,forced=None):
    p=p or StructuredParams()
    probability,events=np.asarray(probability,float),np.asarray(events,float)
    if (probability.ndim!=2 or probability.shape[1]!=3 or len(probability)==0
            or events.shape!=probability.shape or fps<=0 or p.refinement_frames<0
            or not np.isfinite(probability).all() or not np.isfinite(events).all()
            or np.any(probability<0) or not np.allclose(probability.sum(axis=1),1)
            or np.any((events<0)|(events>1))):
        raise ValueError('Invalid structured probabilities or events')
    forced=np.zeros(len(probability),bool) if forced is None else np.asarray(forced,bool)
    if forced.shape!=(len(probability),):
        raise ValueError('Constraint length differs from probabilities')
    fact_edges=np.flatnonzero(np.diff(forced))+1
    edge=np.unique(np.r_[*(candidates(probability,events[:,k],p) for k in range(3)),fact_edges])
    if len(edge)>p.max_candidates:
        raise ValueError('Too many structured boundary candidates')
    coarse=solve(probability,events,fps,p,durations,forced,edge)
    if not p.refinement_frames:
        return coarse
    return refine(probability,events,fps,p,durations,forced,coarse)
