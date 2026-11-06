"""Fixed-capacity, exactly team-equivariant factorized boosted classifiers."""
from dataclasses import asdict, dataclass
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from learned_possession.model import boundary_targets
from .features import swap


@dataclass(frozen=True)
class Config:
    max_iter:int = 250
    learning_rate:float = .05
    max_leaf_nodes:int = 7
    min_samples_leaf:int = 50
    l2_regularization:float = 10.
    early_stopping:bool = False
    random_state:int = 0


def fit(features, codes, config=Config(), step=5, threads=4):
    codes=np.asarray(codes)
    if codes.shape != (len(features.x),) or not np.isin(codes,range(4)).all():
        raise ValueError('Invalid training timeline')
    if step < 1:raise ValueError('Positive training sample step required')
    sample = np.zeros(len(codes),bool)
    for _,first,last in features.halves:sample[first:last+1:step]=True
    sample &= (codes!=3) & features.graded_mask()
    y,usable=boundary_targets(codes,3)
    masks = {'dead':sample, 'owner':sample&(codes<2), 'boundary':sample&usable}
    targets = {'dead':(codes==2).astype(np.int8), 'owner':(codes==0).astype(np.int8), 'boundary':y}
    result={'names':features.names,'fps':features.fps,'config':asdict(config),'step':step,'counts':{}}
    with threadpool_limits(limits=threads):
        for name,selected in masks.items():
            x=features.x[selected];t=targets[name][selected]
            flipped=1-t if name=='owner' else t
            if set(t.tolist())!={0,1}:raise ValueError('Need both binary targets for '+name)
            augmented=np.concatenate([x,swap(x,features.names)])
            target=np.r_[t,flipped]
            result[name]=HistGradientBoostingClassifier(**asdict(config)).fit(augmented,target)
            result['counts'][name]=dict(frames=int(selected.sum()),positive=int(t.sum()),augmented=len(target))
    return result


def predict(bundle, features, threads=4):
    if tuple(bundle['names'])!=features.names or bundle['fps']!=features.fps:
        raise ValueError('Boosted input schema changed')
    p=np.full((len(features.x),3),1/3,np.float32);q=np.zeros(len(features.x),np.float32)
    with threadpool_limits(limits=threads):
        for first in range(0,len(features.x),8192):
            stop=min(len(features.x),first+8192);x=features.x[first:stop];flipped=swap(x,features.names)
            estimates={}
            for name in ('dead','owner','boundary'):
                a=bundle[name].predict_proba(x)[:,1];b=bundle[name].predict_proba(flipped)[:,1]
                estimates[name]=(a + (1-b if name=='owner' else b))/2
            dead,home=estimates['dead'],estimates['owner']
            p[first:stop]=np.column_stack([(1-dead)*home,(1-dead)*(1-home),dead])
            q[first:stop]=estimates['boundary']
    return p,q
