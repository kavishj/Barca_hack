"""Portable frozen visual observations and training-only shared-team PCA."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


@dataclass
class VisualCache:
    centers:np.ndarray
    half_ids:np.ndarray
    embeddings:np.ndarray # centers, wide/home/away, features
    availability:np.ndarray
    halves:tuple
    fps:int
    provenance:dict

    def __post_init__(self):
        self.halves=tuple(map(tuple,self.halves))
        n=len(self.centers)
        if (self.embeddings.ndim!=3 or self.embeddings.shape[:2]!=(n,3) or self.availability.shape!=(n,3)
            or self.half_ids.shape!=(n,) or not np.isfinite(self.embeddings).all()
            or not np.isfinite(self.availability).all() or np.any((self.availability<0)|(self.availability>1))):
            raise ValueError('Invalid frozen visual cache')
        for half,first,last in self.halves:
            c=self.centers[self.half_ids==half]
            if not len(c) or c[0]!=first or c[-1]!=last or np.any(np.diff(c)<=0):
                raise ValueError('Visual centers must cover each half in order')

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,centers=self.centers,half_ids=self.half_ids,embeddings=self.embeddings,
            availability=self.availability,metadata=np.array(json.dumps(dict(halves=self.halves,fps=self.fps,provenance=self.provenance))))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            meta=json.loads(str(d['metadata']))
            return cls(*(d[k] for k in ('centers','half_ids','embeddings','availability')),
                meta['halves'],meta['fps'],meta['provenance'])


@dataclass
class VisualNormalizer:
    means:np.ndarray
    components:np.ndarray
    scales:np.ndarray
    ranks:np.ndarray

    @classmethod
    def fit(cls,cache,selected,components=16):
        take=np.asarray(selected,bool)[cache.centers]
        means=[];vectors=[];scales=[];ranks=[]
        for views in ((0,),(1,2)):
            data=cache.embeddings[take][:,views].reshape(-1,cache.embeddings.shape[-1])
            valid=cache.availability[take][:,views].reshape(-1)>=.75
            data=data[valid]
            rank=min(components,len(data),cache.embeddings.shape[-1])
            mean=np.zeros(cache.embeddings.shape[-1],np.float32)
            matrix=np.zeros((components,cache.embeddings.shape[-1]),np.float32)
            scale=np.ones(components,np.float32)
            if rank>0:
                with threadpool_limits(limits=4):
                    pca=PCA(n_components=rank,svd_solver='randomized',random_state=0).fit(data)
                    matrix[:rank]=pca.components_;mean=pca.mean_
                    scale[:rank]=np.maximum(np.std((data-mean)@pca.components_.T,axis=0),1e-5)
            means.append(mean);vectors.append(matrix);scales.append(scale);ranks.append(rank)
        return cls(np.array(means),np.array(vectors),np.array(scales),np.array(ranks))

    def transform(self,cache,length):
        k=self.components.shape[1]
        transformed=np.zeros((len(cache.centers),3,k+1),np.float32)
        with threadpool_limits(limits=4):
            for view in (0,1,2):
                group=int(view>0)
                x=(cache.embeddings[:,view]-self.means[group])@self.components[group].T/self.scales[group]
                availability=cache.availability[:,view]
                transformed[:,view,:k]=np.clip(x,-8,8)*(availability>0)[:,None]
                transformed[:,view,k]=availability
        dense=np.zeros((length,3,k+1),np.float32)
        for half,first,last in cache.halves:
            take=cache.half_ids==half;frames=np.arange(first,last+1)
            for view in (0,1,2):
                for column in range(k+1):
                    dense[first:last+1,view,column]=np.interp(frames,cache.centers[take],transformed[take,view,column])
        return dense[:,0],dense[:,1:]

    def save(self,path):np.savez_compressed(path,means=self.means,components=self.components,scales=self.scales,ranks=self.ranks)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:return cls(*(d[k] for k in ('means','components','scales','ranks')))
