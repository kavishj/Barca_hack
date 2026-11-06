"""Live/team factorisation and stop, restart, turnover heads on one TCN."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from learned_possession.model import boundary_targets, near
from temporal_possession.network import ResidualBlock, context, deterministic, epoch_tiles

EVENTS = ('stop','restart','turnover')


def targets(codes,radius=3):
    _,usable = boundary_targets(codes,radius)
    before,after = codes[:-1],codes[1:]
    known = (before!=3)&(after!=3)
    events = np.zeros((len(codes),3),np.float32)
    conditions = [known&(before<2)&(after==2), known&(before==2)&(after<2),
                  known&(before<2)&(after<2)&(before!=after)]
    for k,condition in enumerate(conditions):
        markers=np.r_[False,condition]
        events[:,k]=near(markers,radius)
    return events,usable


class FactorizedNet(nn.Module):
    def __init__(self,input_channels,config):
        super().__init__()
        self.config=config
        self.input=nn.Conv1d(input_channels,config.width,1)
        self.blocks=nn.Sequential(*(ResidualBlock(config.width,d,config.dropout) for d in config.dilations))
        self.live=nn.Conv1d(config.width,1,1)
        self.home=nn.Conv1d(config.width,1,1)
        self.events=nn.Conv1d(config.width,3,1)

    def forward(self,x):
        h=self.blocks(F.gelu(self.input(x)))
        return self.live(h)[:,0],self.home(h)[:,0],self.events(h)


def probabilities(live,home):
    live,home=torch.sigmoid(live),torch.sigmoid(home)
    return torch.stack([live*home,live*(1-home),1-live],dim=1)


def train_seed(x,codes,selected,halves,config,seed,device):
    deterministic(seed)
    rng=np.random.default_rng(seed)
    by,bm=targets(codes)
    positive=by[bm].sum(axis=0)
    negative=bm.sum()-positive
    if np.any(positive<=0) or np.any(negative<=0):
        raise ValueError('Each typed event needs positive and negative training examples')
    weights=np.minimum(20,negative/positive).astype(np.float32)
    model=FactorizedNet(x.shape[1],config).to(device)
    with torch.no_grad():
        model.events.bias.copy_(torch.from_numpy(np.log(weights*positive/negative)).to(device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs)
    event_weight=torch.tensor(weights,device=device)
    losses=[]
    for epoch in range(config.epochs):
        model.train()
        tiles=epoch_tiles(selected,halves,config.core_frames,rng)
        values=[]
        for start in range(0,len(tiles),config.batch_size):
            data,ys,masks,event_ys,event_masks=[],[],[],[],[]
            for frame,first,stop,run_first,run_stop in tiles[start:start+config.batch_size]:
                data.append(context(x,frame,config.core_frames,config.radius,first,stop))
                indices=np.arange(frame,frame+config.core_frames)
                inside=(indices>=run_first)&(indices<run_stop)
                indices=np.clip(indices,0,len(codes)-1)
                ys.append(codes[indices])
                masks.append(inside&selected[indices])
                event_ys.append(by[indices])
                event_masks.append(inside&bm[indices])
            data=torch.from_numpy(np.stack(data)).to(device)
            y=torch.from_numpy(np.stack(ys).astype(np.int64)).to(device)
            mask=torch.from_numpy(np.stack(masks)).to(device)
            ey=torch.from_numpy(np.stack(event_ys)).to(device)
            em=torch.from_numpy(np.stack(event_masks)).to(device)
            region=slice(config.radius,config.radius+config.core_frames)
            live,home,event=model(data)
            live,home,event=live[:,region],home[:,region],event[:,:,region].transpose(1,2)
            # Product distribution, with team loss only on live frames: equal to
            # three-class NLL, while explicit factors support physical decoding.
            loss=F.binary_cross_entropy_with_logits(live[mask],(y[mask]<2).float())
            alive=mask&(y<2)
            if alive.any():
                loss=loss+alive.sum()/mask.sum()*F.binary_cross_entropy_with_logits(
                    home[alive],(y[alive]==0).float())
            if em.any():
                loss=loss+0.25*F.binary_cross_entropy_with_logits(event[em],ey[em],pos_weight=event_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            values.append(float(loss.detach()))
        scheduler.step()
        value=float(np.mean(values))
        if not np.isfinite(value):
            raise ValueError('Non-finite factorised training loss')
        losses.append(value)
        if epoch==0 or (epoch+1)%10==0:
            print(f'factorised seed {seed} epoch {epoch+1}/{config.epochs} loss {value:.5f}',flush=True)
    model.eval()
    return model,dict(seed=seed,training_loss=losses,event_positive_weights=weights.tolist(),
                      event_positive_frames=positive.tolist(),event_mask_frames=int(bm.sum()),
                      prior_correction='subtract log(positive weight) for each event',
                      parameters=sum(p.numel() for p in model.parameters()))


def infer_seed(model,x,halves,weights,device,core=2048,batch_size=4):
    probability=np.full((len(x),3),1/3,np.float32)
    events=np.zeros((len(x),3),np.float32)
    radius=model.config.radius
    tiles=[(start,first,last+1) for _,first,last in halves for start in range(first,last+1,core)]
    model.eval()
    with torch.inference_mode():
        for offset in range(0,len(tiles),batch_size):
            batch=tiles[offset:offset+batch_size]
            data=np.stack([context(x,start,core,radius,first,stop) for start,first,stop in batch])
            live,home,event=model(torch.from_numpy(data).to(device))
            region=slice(radius,radius+core)
            p=probabilities(live[:,region],home[:,region]).transpose(1,2).cpu().numpy()
            q=torch.sigmoid(event[:,:,region]-torch.tensor(np.log(weights),device=device)[None,:,None])
            q=q.transpose(1,2).cpu().numpy()
            for i,(start,_,stop) in enumerate(batch):
                count=min(core,stop-start)
                probability[start:start+count]=p[i,:count]
                events[start:start+count]=q[i,:count]
    return probability,events
