"""The original graph training objective with optional frozen-video observations."""
import numpy as np
import torch
from torch.nn import functional as F
from learned_possession.model import boundary_targets
from temporal_possession.network import deterministic,known_runs
from relational_possession.engine import tensors
from .model import VisionNet


def train_seed(global_x,pair_x,graph,codes,selected,config,seed,device,enabled=True):
    deterministic(seed);rng=np.random.default_rng(seed)
    model=VisionNet(global_x.shape[1],pair_x.shape[2],graph.nodes.shape[2],config,
                    graph.global_visual.shape[1],enabled).to(device)
    boundary,usable=boundary_targets(codes,3)
    positive=float(boundary[usable].sum());negative=float(usable.sum())-positive
    if min(positive,negative)<=0:raise ValueError('Training needs known interior boundaries and non-boundaries')
    weight=min(20.,negative/positive)
    with torch.no_grad():model.boundary.bias.fill_(np.log(weight*positive/negative))
    windows=known_runs(selected)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs)
    losses=[]
    for epoch in range(config.epochs):
        model.train();values=[]
        for window in rng.permutation(len(windows)):
            first,stop=windows[window]
            blocks=[h for h in graph.halves if h[1]<=first and stop<=h[2]+1]
            if len(blocks)!=1:raise ValueError('Known training interval crosses halftime')
            half,left,last=blocks[0]
            a=max(left,first-max(config.radius,config.warm_seconds*graph.fps))
            b=min(last+1,stop+config.radius)
            gx=torch.from_numpy(global_x[a:b]).to(device);px=torch.from_numpy(pair_x[a:b]).to(device)
            logits,boundary_logits,_=model(gx,px,tensors(graph,a,b,half,device))
            core=slice(first-a,stop-a)
            classes=torch.as_tensor(codes[first:stop].astype(np.int64),device=device)
            fraction=(stop-first)/int(selected.sum())*len(windows)
            loss=fraction*F.cross_entropy(logits[core],classes)
            mask=torch.as_tensor(usable[first:stop],device=device)
            if mask.any():
                by=torch.as_tensor(boundary[first:stop].astype(np.float32),device=device)
                loss=loss+.25*fraction*F.binary_cross_entropy_with_logits(
                    boundary_logits[core][mask],by[mask],pos_weight=torch.tensor(weight,device=device))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            values.append(float(loss.detach()))
        scheduler.step();average=float(np.mean(values))
        if not np.isfinite(average):raise ValueError('Non-finite visual-fusion training loss')
        losses.append(average)
        if epoch==0 or (epoch+1)%10==0:
            print(f'Vision fusion seed {seed} epoch {epoch+1}/{config.epochs} train loss {average:.5f}',flush=True)
    model.eval()
    return model,dict(seed=seed,training_loss=losses,boundary_positive_weight=weight,
        total_registered_parameters=sum(p.numel() for p in model.parameters()),
        known_frames=int(selected.sum()),known_windows=len(windows),video_encoder_frozen=True,
        state_loss='Unchanged graph CE and boundary BCE; natural frame frequency')
