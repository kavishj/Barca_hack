"""Preserve the baseline's parameters/RNG and add zero-initialized event input."""
import torch
from torch import nn
from torch.nn import functional as F
from relational_possession.model import RelationalNet


class HistoryLinear(nn.Module):
    def __init__(self,base,extra,enabled=True):
        super().__init__();self.weight=base.weight;self.bias=base.bias
        self.in_features=base.in_features;self.out_features=base.out_features
        self.enabled=enabled
        with torch.random.fork_rng(devices=[]):self.history=nn.Linear(extra,base.out_features,bias=False)
        nn.init.zeros_(self.history.weight)
    def forward(self,x):
        result=F.linear(x[...,:self.in_features],self.weight,self.bias)
        if self.enabled:result=result+self.history(x[...,self.in_features:])
        return result


class RecoveryNet(RelationalNet):
    def __init__(self,global_features,pair_features,node_features,config,history=True):
        if node_features<=15:raise ValueError('Recovery graph requires explicit observation history')
        super().__init__(global_features,pair_features,15,config,'graph_tcn')
        self.spatial.input=HistoryLinear(self.spatial.input,node_features-15,history)
