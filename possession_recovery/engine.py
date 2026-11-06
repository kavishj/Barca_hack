"""Original graph training with explicitly controlled history and spell weighting."""
import numpy as np
import torch
from torch.nn import functional as F
from learned_possession.model import boundary_targets
from temporal_possession.network import deterministic,known_runs
from relational_possession.engine import tensors
from .model import RecoveryNet
from .loss import spell_weights


def train_seed(gx,px,graph,codes,selected,config,seed,device,history,balanced):
    deterministic(seed);rng=np.random.default_rng(seed)
    model=RecoveryNet(gx.shape[1],px.shape[2],graph.nodes.shape[2],config,history).to(device)
    boundary,usable=boundary_targets(codes,3)
    positive=float(boundary[usable].sum());negative=float(usable.sum())-positive
    if min(positive,negative)<=0:raise ValueError('Training needs known interior boundaries')
    pos_weight=min(20.,negative/positive);prior=positive/(positive+negative)
    with torch.no_grad():model.boundary.bias.fill_(np.log(pos_weight*positive/negative))
    weights,weight_info=spell_weights(codes)
    if not balanced:weights=np.where(selected,1.,0.).astype(np.float32)
    windows=known_runs(selected)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs)
    losses=[]
    for epoch in range(config.epochs):
        model.train();values=[]
        for window in rng.permutation(len(windows)):
            first,stop=windows[window]
            half,left,last=next(h for h in graph.halves if h[1]<=first and stop<=h[2]+1)
            a=max(left,first-max(config.radius,config.warm_seconds*graph.fps));b=min(last+1,stop+config.radius)
            logits,bl,_=model(torch.from_numpy(gx[a:b]).to(device),torch.from_numpy(px[a:b]).to(device),tensors(graph,a,b,half,device))
            core=slice(first-a,stop-a);classes=torch.as_tensor(codes[first:stop].astype(np.int64),device=device)
            fraction=(stop-first)/int(selected.sum())*len(windows)
            if balanced:
                per_frame=F.cross_entropy(logits[core],classes,reduction='none')
                loss=fraction*(per_frame*torch.as_tensor(weights[first:stop],device=device)).mean()
            else:
                loss=fraction*F.cross_entropy(logits[core],classes)
            mask=torch.as_tensor(usable[first:stop],device=device)
            if mask.any():
                by=torch.as_tensor(boundary[first:stop].astype(np.float32),device=device)
                loss=loss+.25*fraction*F.binary_cross_entropy_with_logits(bl[core][mask],by[mask],pos_weight=torch.tensor(pos_weight,device=device))
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            values.append(float(loss.detach()))
        scheduler.step();average=float(np.mean(values))
        if not np.isfinite(average):raise ValueError('Non-finite recovery loss')
        losses.append(average)
        if epoch==0 or (epoch+1)%10==0:
            print(f'Recovery history={history} balanced={balanced} seed {seed} epoch {epoch+1}/{config.epochs} loss {average:.5f}',flush=True)
    model.eval()
    return model,dict(seed=seed,training_loss=losses,boundary_positive_weight=pos_weight,boundary_training_prior=prior,
        known_frames=int(selected.sum()),known_windows=len(windows),balanced=balanced,history=history,
        weight_info=weight_info,total_registered_parameters=sum(p.numel() for p in model.parameters()))
