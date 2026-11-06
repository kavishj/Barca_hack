import numpy as np
import torch
from relational_possession.data import SymmetricNormalizer


class Reflections:
    def __init__(self,normalizer,seed):
        self.normalizer=normalizer
        self.rng=np.random.default_rng(seed+78173)
        self.y_columns=[]
        names=normalizer.audio_normalizer.names
        for i,raw_index in enumerate(normalizer.global_indices):
            if names[raw_index] in ('ball_y','ball_vy'):
                self.y_columns.append((i,float(normalizer.global_mean[i]/normalizer.global_scale[i])))

    def transform(self,gx,px,graph,flip_x,flip_y):
        observed=dict(graph)
        if flip_x or flip_y:
            nodes=graph['nodes'].clone()
            if flip_x:nodes[:,:,[0,2,5,8]]*=-1
            if flip_y:nodes[:,:,[1,3,6,9]]*=-1
            observed['nodes']=nodes
        if flip_y:
            gx=gx.clone();dimension=len(self.normalizer.global_mean)
            for index,offset in self.y_columns:
                missing=gx[:,dimension+index]!=0
                gx[:,index]=torch.where(missing,torch.zeros_like(gx[:,index]),
                    torch.clamp(-gx[:,index]-2*offset,-8,8))
        return gx,px,observed

    def __call__(self,gx,px,graph):
        flip_x,flip_y=self.rng.integers(0,2,size=2)
        return self.transform(gx,px,graph,bool(flip_x),bool(flip_y))
