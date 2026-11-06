"""Zero-initialized frozen-vision fusion into the winning relational model."""
import torch
from torch import nn
from torch.nn import functional as F
from relational_possession.model import RelationalNet


class VisionNet(RelationalNet):
    def __init__(self,global_features,pair_features,node_features,config,vision_features,enabled=True):
        super().__init__(global_features,pair_features,node_features,config,'graph_tcn')
        self.vision_enabled=enabled
        # Preserve the old model's initialization and subsequent dropout RNG.
        with torch.random.fork_rng(devices=[]):
            self.visual_global=nn.Linear(vision_features,config.width)
            self.visual_pair=nn.Linear(vision_features,config.width)
        for layer in (self.visual_global,self.visual_pair):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,global_x,pair_x,graph,graph_chunk=None):
        global_logit=self.global_project(global_x)
        pair_logit=self.pair_project(pair_x)
        if self.vision_enabled:
            vg,vp=graph['visual_global'],graph['visual_pair']
            global_logit=global_logit+self.visual_global(vg)*vg[:,-1:]
            pair_logit=pair_logit+self.visual_pair(vp)*vp[:,:,-1:]
        global_h,pair_h=F.gelu(global_logit),F.gelu(pair_logit)
        coarse=self.graph_features(graph,graph_chunk)
        left,right,weight=graph['left'],graph['right'],graph['weight'][:,None,None]
        spatial=coarse[left]*(1-weight)+coarse[right]*weight
        z=torch.cat([global_h[:,None].expand(-1,2,-1),pair_h,spatial,torch.zeros_like(spatial)],dim=-1)
        h=self.blocks(F.gelu(self.input(z.permute(1,2,0)))).permute(2,0,1)
        invariant=torch.cat([h.mean(dim=1),(h[:,0]-h[:,1]).abs()],dim=-1)
        latent=self.owner_substates(h)
        logits=torch.stack([latent[:,0].logsumexp(dim=1),latent[:,1].logsumexp(dim=1),self.dead(invariant)[:,0]],dim=1)
        return logits,self.boundary(invariant)[:,0],F.softplus(self.span(h))*5


class VisualGraph:
    def __init__(self,graph,global_visual,pair_visual):
        self.base=graph;self.global_visual=global_visual;self.pair_visual=pair_visual
        self.halves,self.fps,self.nodes=graph.halves,graph.fps,graph.nodes
    def window(self,first,stop,half):
        result=self.base.window(first,stop,half)
        result['visual_global']=self.global_visual[first:stop]
        result['visual_pair']=self.pair_visual[first:stop]
        return result
