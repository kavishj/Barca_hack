"""Spatial ownership pretraining on external passes only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from relational_possession.model import SpatialEncoder,Config
from temporal_possession.network import deterministic
from temporal_possession.audio import sha
from temporal_possession.experiment import save_json


def augment(nodes,roles,valid,positioned,player_drop=.1,ball_drop=.2):
    nodes=nodes.clone();valid=valid.clone();positioned=positioned.clone()
    n=len(nodes)
    fx=torch.where(torch.rand(n,device=nodes.device)<.5,-1.,1.)
    fy=torch.where(torch.rand(n,device=nodes.device)<.5,-1.,1.)
    nodes[:,:,[0,2,5,8]]*=fx[:,None,None];nodes[:,:,[1,3,6,9]]*=fy[:,None,None]
    dropped=(torch.rand(valid.shape,device=nodes.device)<player_drop)&(roles<2)
    valid[dropped]=False;positioned[dropped]=False;nodes[dropped]=0
    missing=torch.rand(n,device=nodes.device)<ball_drop
    nodes[missing,:,5:10]=0;nodes[missing,:,14]=0
    nodes[missing,-1,:10]=0;nodes[missing,-1,12:15]=0;positioned[missing,-1]=False
    return nodes,roles,valid,positioned


class Owner(nn.Module):
    def __init__(self):
        super().__init__();self.spatial=SpatialEncoder(15,Config());self.owner=nn.Linear(32,1)
    def forward(self,*data):return self.owner(self.spatial(*data))[:,:,0]


def run(args):
    out=Path(args.out)
    if out.exists():raise ValueError('Use a fresh external pretraining directory')
    out.mkdir(parents=True);deterministic(0);rng=np.random.default_rng(0)
    with np.load(args.samples,allow_pickle=False) as d:
        metadata=json.loads(str(d['metadata']))
        if metadata['target_match_labels_used']:raise ValueError('External samples contain target supervision')
        data=[torch.as_tensor(d[name],device=args.device) for name in ('nodes','roles','valid','positioned')]
        labels=torch.as_tensor(d['labels'].astype(np.int64),device=args.device)
    if set(labels.cpu().numpy().tolist())!={0,1}:raise ValueError('Pretraining uses only H/A pass supervision')
    plan=dict(kind='external_spatial_pass_pretraining',seed=0,epochs=20,batch_size=128,learning_rate=.001,
        weight_decay=.001,player_dropout=.1,ball_dropout=.2,reflections=True,samples_sha256=sha(args.samples),metadata=metadata,
        target_match_labels_used=False,early_stopping=False)
    save_json(out/'experiment.json',plan)
    model=Owner().to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20);losses=[]
    for epoch in range(20):
        model.train();total=0.
        order=rng.permutation(len(labels))
        for first in range(0,len(order),128):
            selected=torch.as_tensor(order[first:first+128],device=args.device)
            observed=augment(*(value[selected] for value in data))
            logits=model(*observed);loss=F.cross_entropy(logits,labels[selected])
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(selected)
        schedule.step();losses.append(total/len(labels))
        if not np.isfinite(losses[-1]):raise ValueError('Nonfinite external pretraining loss')
        print('EXTERNAL epoch',epoch+1,'/20 loss',losses[-1],flush=True)
    checkpoint=out/'spatial.pt';torch.save(model.spatial.state_dict(),checkpoint)
    save_json(out/'fit.json',dict(encoder_sha256=sha(checkpoint),training_loss=losses,samples=len(labels),
        encoder_parameters=sum(p.numel() for p in model.spatial.parameters()),target_match_labels_used=False))
    print('EXTERNAL PRETRAIN COMPLETE',sha(checkpoint),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('samples','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu');run(parser.parse_args())
