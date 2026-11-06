"""Shared player-ball graph, team-equivariant temporal head and gated memory."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from temporal_possession.network import ResidualBlock

ARMS=('symmetric_tcn','graph_tcn','graph_memory','graph_memory_span')


@dataclass(frozen=True)
class Config:
    width:int=32
    heads:int=4
    graph_layers:int=2
    dilations:tuple=(1,2,4,8,16,32)
    dropout:float=.15
    epochs:int=60
    learning_rate:float=.001
    weight_decay:float=.001
    warm_seconds:int=60
    span_weight:float=.25
    iou_weight:float=.25

    def __post_init__(self):
        if self.width<1 or self.heads<1 or self.width%self.heads or self.graph_layers<1 or self.epochs<1:
            raise ValueError('Invalid relational architecture settings')

    @property
    def radius(self):
        return 2*sum(self.dilations)


class GraphLayer(nn.Module):
    def __init__(self,width,heads,dropout):
        super().__init__()
        self.heads=heads
        self.qkv=nn.Linear(width,width*3)
        self.output=nn.Linear(width,width)
        self.distance_scale=nn.Parameter(torch.zeros(heads))
        self.same_team=nn.Parameter(torch.zeros(heads))
        self.norm1,self.norm2=nn.LayerNorm(width),nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(width*2,width))

    def forward(self,h,valid,distance,same):
        t,n,d=h.shape
        q,k,v=self.qkv(h).reshape(t,n,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        score=q@k.transpose(-1,-2)/math.sqrt(d//self.heads)
        score=score-F.softplus(self.distance_scale)[None,:,None,None]*distance[:,None]
        score=score+self.same_team[None,:,None,None]*same[:,None]
        score=score.masked_fill(~valid[:,None,None,:],-1e4)
        z=(score.softmax(dim=-1)@v).transpose(1,2).reshape(t,n,d)
        h=self.norm1(h+self.output(z))
        return self.norm2(h+self.ff(h))*valid[:,:,None]


class SpatialEncoder(nn.Module):
    def __init__(self,features,config):
        super().__init__()
        self.input=nn.Linear(features,config.width)
        self.layers=nn.ModuleList(GraphLayer(config.width,config.heads,config.dropout) for _ in range(config.graph_layers))
        self.pool=nn.Sequential(nn.Linear(2*config.width,config.width),nn.GELU(),nn.Linear(config.width,1))

    def forward(self,nodes,roles,valid,positioned):
        # Team identifiers affect only equality relations and shared pooling.
        position=nodes[:,:,:2]*nodes.new_tensor([5.265,3.4])
        distance=torch.cdist(position,position)
        distance=distance*(positioned[:,:,None]&positioned[:,None,:])
        same=((roles[:,:,None]==roles[:,None,:])&(roles[:,:,None]<2)&(roles[:,None,:]<2)).to(nodes.dtype)
        h=F.gelu(self.input(nodes))*valid[:,:,None]
        for layer in self.layers:
            h=layer(h,valid,distance,same)
        ball=h[:,-1:, :].expand_as(h)
        score=self.pool(torch.cat([h,ball],dim=-1))[:,:,0]
        teams=[]
        for club in (0,1):
            mask=valid&(roles==club)
            weight=score.masked_fill(~mask,-1e4).softmax(dim=1)*mask
            weight=weight/weight.sum(dim=1,keepdim=True).clamp_min(1e-6)
            teams.append((h*weight[:,:,None]).sum(dim=1))
        return torch.stack(teams,dim=1)


class ObservationMemory(nn.Module):
    """Causal diagonal state space; uncertainty slows ownership updates.

    Short associative blocks avoid an expensive per-frame Python GRU and keep
    products away from underflow. State is reset explicitly at each half/window.
    """
    def __init__(self,inputs,width):
        super().__init__()
        self.candidate=nn.Linear(inputs,width)
        self.retention=nn.Linear(inputs,width)
        self.register_buffer('lower',torch.tril(torch.ones(32,32)),persistent=False)

    def forward(self,x,confidence,initial=None,block=32):
        # x: teams,time,channels; the same parameters serve both teams.
        if not 1<=block<=32:
            raise ValueError('Memory block must contain 1..32 steps')
        candidate=torch.tanh(self.candidate(x))
        retention=.5+.4995*torch.sigmoid(self.retention(x))
        update=(1-retention)*(.05+.95*confidence[None,:,None])
        alpha=1-update
        h=torch.zeros_like(candidate[:,0]) if initial is None else initial
        outputs=[]
        for start in range(0,x.shape[1],block):
            a=alpha[:,start:start+block]
            b=update[:,start:start+block]*candidate[:,start:start+block]
            lower=self.lower[:a.shape[1],:a.shape[1]].to(dtype=a.dtype)
            # Triangular matmul keeps the scan deterministic on CUDA builds
            # where floating cumsum/cumprod have no deterministic backward.
            product=torch.exp((torch.log(a).transpose(1,2)@lower.T).transpose(1,2))
            prefix=((b/product).transpose(1,2)@lower.T).transpose(1,2)
            state=product*(h[:,None]+prefix)
            h=state[:,-1]
            outputs.append(state)
        return torch.cat(outputs,dim=1),h


class RelationalNet(nn.Module):
    def __init__(self,global_features,pair_features,node_features,config,arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown relational ablation')
        self.config,self.arm=config,arm
        d=config.width
        self.spatial=SpatialEncoder(node_features,config)
        self.global_project=nn.Linear(global_features,d)
        self.pair_project=nn.Linear(pair_features,d)
        self.memory=ObservationMemory(d*3,d)
        self.input=nn.Conv1d(d*4,d,1)
        self.blocks=nn.Sequential(*(ResidualBlock(d,k,config.dropout) for k in config.dilations))
        self.owner_substates=nn.Linear(d,2)
        self.dead=nn.Linear(d*2,1)
        self.boundary=nn.Linear(d*2,1)
        self.span=nn.Linear(d,2)
        nn.init.constant_(self.dead.bias,math.log(2))

    def graph_features(self,graph,chunk=None):
        if self.arm=='symmetric_tcn':
            return graph['nodes'].new_zeros((len(graph['nodes']),2,self.config.width))
        if chunk is None:
            return self.spatial(*(graph[key] for key in ('nodes','roles','valid','positioned')))
        return torch.cat([self.spatial(*(graph[key][a:a+chunk] for key in ('nodes','roles','valid','positioned')))
                          for a in range(0,len(graph['nodes']),chunk)])

    def forward(self,global_x,pair_x,graph,graph_chunk=None):
        global_h=F.gelu(self.global_project(global_x))
        pair_h=F.gelu(self.pair_project(pair_x))
        coarse=self.graph_features(graph,graph_chunk)
        if self.arm in ('graph_memory','graph_memory_span'):
            at=graph['coarse_to_fine']
            context=torch.cat([coarse,pair_h[at],global_h[at,None].expand(-1,2,-1)],dim=-1).transpose(0,1)
            memory,_=self.memory(context,graph['confidence'])
            memory=memory.transpose(0,1)
        else:
            memory=torch.zeros_like(coarse)
        left,right,weight=graph['left'],graph['right'],graph['weight'][:,None,None]
        spatial=coarse[left]*(1-weight)+coarse[right]*weight
        memory=memory[left]*(1-weight)+memory[right]*weight
        z=torch.cat([global_h[:,None].expand(-1,2,-1),pair_h,spatial,memory],dim=-1)
        h=self.blocks(F.gelu(self.input(z.permute(1,2,0)))).permute(2,0,1)
        invariant=torch.cat([h.mean(dim=1),(h[:,0]-h[:,1]).abs()],dim=-1)
        latent=self.owner_substates(h)
        # Two latent substates per owner, not independently verified action labels.
        logits=torch.stack([latent[:,0].logsumexp(dim=1),latent[:,1].logsumexp(dim=1),self.dead(invariant)[:,0]],dim=1)
        boundary=self.boundary(invariant)[:,0]
        spans=F.softplus(self.span(h))*5  # positive left/right distances in seconds
        return logits,boundary,spans
