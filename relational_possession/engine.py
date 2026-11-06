"""Full-window supervision, masked span targets and label-free half inference."""
from dataclasses import asdict
import numpy as np
import torch
from torch.nn import functional as F

from learned_possession.model import boundary_targets
from temporal_possession.network import deterministic,known_runs
from .model import Config,RelationalNet
from .loss import span_targets,span_loss


def tensors(graph,first,stop,half,device):
    return {key:torch.as_tensor(value,device=device) for key,value in graph.window(first,stop,half).items()}


def train_seed(global_x,pair_x,graph,codes,selected,config,arm,seed,device,model_factory=RelationalNet,observation_transform=None):
    deterministic(seed)
    rng=np.random.default_rng(seed)
    model=model_factory(global_x.shape[1],pair_x.shape[2],graph.nodes.shape[2],config,arm).to(device)
    boundary,usable=boundary_targets(codes,3)
    positive=float(boundary[usable].sum())
    negative=float(usable.sum())-positive
    if min(positive,negative)<=0:
        raise ValueError('Training requires interior changes and non-changes')
    weight=min(20.,negative/positive)
    with torch.no_grad():
        model.boundary.bias.fill_(np.log(weight*positive/negative))
    sy,sm,sw=span_targets(codes,graph.fps)
    windows=known_runs(selected)
    endpoint_total=float((sw[:,None]*sm).sum())
    complete_total=float(sw[sm.all(axis=1)].sum())
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs)
    losses=[]
    for epoch in range(config.epochs):
        model.train()
        values=[]
        for window in rng.permutation(len(windows)):
            first,stop=windows[window]
            blocks=[h for h in graph.halves if h[1]<=first and stop<=h[2]+1]
            if len(blocks)!=1:
                raise ValueError('A training run crosses halftime')
            half,left,last=blocks[0]
            a=max(left,first-max(config.radius,config.warm_seconds*graph.fps))
            b=min(last+1,stop+config.radius)
            gx=torch.from_numpy(global_x[a:b]).to(device)
            px=torch.from_numpy(pair_x[a:b]).to(device)
            data=tensors(graph,a,b,half,device)
            if observation_transform is not None:
                gx,px,data=observation_transform(gx,px,data)
            logits,boundary_logits,spans=model(gx,px,data)
            core=slice(first-a,stop-a)
            classes=torch.as_tensor(codes[first:stop].astype(np.int64),device=device)
            fraction=(stop-first)/int(selected.sum())*len(windows)
            loss=fraction*F.cross_entropy(logits[core],classes)
            mask=torch.as_tensor(usable[first:stop],device=device)
            if mask.any():
                by=torch.as_tensor(boundary[first:stop].astype(np.float32),device=device)
                loss=loss+.25*fraction*F.binary_cross_entropy_with_logits(
                    boundary_logits[core][mask],by[mask],pos_weight=torch.tensor(weight,device=device))
            if arm=='graph_memory_span':
                target=torch.as_tensor(sy[first:stop],device=device)
                observed=torch.as_tensor(sm[first:stop],device=device)
                weights=torch.as_tensor(sw[first:stop],device=device)
                regression,overlap=span_loss(spans[core],classes,target,observed,weights)
                endpoint_mass=float((sw[first:stop,None]*sm[first:stop]).sum())
                complete_mass=float(sw[first:stop][sm[first:stop].all(axis=1)].sum())
                loss=loss+config.span_weight*regression*endpoint_mass/max(endpoint_total,1e-6)*len(windows)
                loss=loss+config.iou_weight*overlap*complete_mass/max(complete_total,1e-6)*len(windows)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            values.append(float(loss.detach()))
        scheduler.step()
        average=float(np.mean(values))
        if not np.isfinite(average):
            raise ValueError('Non-finite relational training loss')
        losses.append(average)
        if epoch==0 or (epoch+1)%10==0:
            print(f'{arm} seed {seed} epoch {epoch+1}/{config.epochs} train loss {average:.5f}',flush=True)
    model.eval()
    return model,dict(seed=seed,training_loss=losses,boundary_positive_weight=weight,
                      total_registered_parameters=sum(p.numel() for p in model.parameters()),
                      state_loss='Natural frame frequency; shared team parameters',
                      span_loss='Per-possession endpoint regression and IoU; censored ends masked',
                      known_frames=int(selected.sum()),known_windows=len(windows))


def infer_seed(model,global_x,pair_x,graph,weight,device,graph_chunk=512):
    model.eval()
    p=np.full((len(global_x),3),1/3,np.float32)
    q=np.zeros(len(global_x),np.float32)
    with torch.inference_mode():
        for half,first,last in graph.halves:
            gx=torch.from_numpy(global_x[first:last+1]).to(device)
            px=torch.from_numpy(pair_x[first:last+1]).to(device)
            data=tensors(graph,first,last+1,half,device)
            logits,boundary,_=model(gx,px,data,graph_chunk=graph_chunk)
            p[first:last+1]=logits.softmax(dim=1).cpu().numpy()
            q[first:last+1]=torch.sigmoid(boundary-np.log(weight)).cpu().numpy()
    return p,q
