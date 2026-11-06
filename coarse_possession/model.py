from dataclasses import dataclass
import torch
from relational_possession.model import Config as BaseConfig,RelationalNet


@dataclass(frozen=True)
class Config(BaseConfig):
    stride:int=5

    @property
    def radius(self):return super().radius*self.stride


class CoarseNet(RelationalNet):
    """Apply unchanged temporal blocks on sampled observations, then interpolate."""
    def forward(self,global_x,pair_x,graph,graph_chunk=None):
        at=graph['coarse_to_fine']
        identity=torch.arange(len(at),device=at.device)
        coarse=dict(graph,left=identity,right=identity,weight=graph['weight'].new_zeros(len(at)),
                    coarse_to_fine=identity)
        logits,boundary,spans=super().forward(global_x[at],pair_x[at],coarse,graph_chunk)
        left,right,weight=graph['left'],graph['right'],graph['weight']
        def interpolate(values):
            w=weight.reshape((-1,)+(1,)*(values.ndim-1))
            return values[left]*(1-w)+values[right]*w
        return interpolate(logits),interpolate(boundary),interpolate(spans)
