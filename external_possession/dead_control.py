"""Predeclared external H/A/D temporal model against a non-temporal baseline."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import expit,softmax
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from temporal_possession.network import deterministic,known_runs,epoch_tiles
from temporal_possession.audio import sha
from temporal_possession.experiment import save_json
from .temporal_control import ControlNet,random_blocks,block_context,infer


class DeadControlNet(ControlNet):
    states=3
    def __init__(self):
        super().__init__();self.dead=nn.Conv1d(64,1,1)
        nn.init.zeros_(self.dead.weight);nn.init.zeros_(self.dead.bias)

    def forward(self,x,point):
        b,teams,c,t=x.shape
        h=self.blocks(F.gelu(self.input(x.reshape(b*teams,c,t)))).reshape(b,teams,32,t)
        owner=self.owner(h.reshape(b*teams,32,t)).reshape(b,teams,t)+point
        invariant=torch.cat([h.mean(1),(h[:,0]-h[:,1]).abs()],dim=1)
        dead=self.dead(invariant)+torch.logsumexp(owner,dim=1,keepdim=True)
        return torch.cat([owner,dead],dim=1),self.boundary(invariant)[:,0]


def invariant_features(x,point):
    if x.shape!=(len(point),2,35) or point.shape!=(len(x),2):raise ValueError('Wrong external feature schema')
    return np.concatenate([x[:,:,:32].mean(1),np.abs(x[:,0,:32]-x[:,1,:32]),
        np.abs(point[:,0]-point[:,1])[:,None],x[:,0,33:35]],axis=1)


def baseline_fit(x,point,labels):
    features=invariant_features(x,point);selected=labels<3
    mean=features[selected].mean(0);scale=np.maximum(features[selected].std(0),1e-3)
    normalized=np.clip((features[selected]-mean)/scale,-8,8).astype(np.float64)
    with threadpool_limits(limits=4):
        model=LogisticRegression(C=.1,fit_intercept=True,max_iter=2000,tol=1e-8,random_state=0).fit(normalized,(labels[selected]==2).astype(int))
    if model.n_iter_[0]>=2000:raise ValueError('External dead baseline did not converge')
    return dict(mean=mean,scale=scale,coef=model.coef_[0],intercept=model.intercept_[0])


def baseline_probability(x,point,fit):
    z=np.clip((invariant_features(x,point)-fit['mean'])/fit['scale'],-8,8)
    dead=expit(z@fit['coef']+fit['intercept']);home=expit(point[:,0]-point[:,1])
    return np.column_stack([(1-dead)*home,(1-dead)*(1-home),dead])


def boundary_targets(labels,half_ids):
    y=np.zeros(len(labels),np.float32);usable=np.zeros(len(labels),bool)
    for half in np.unique(half_ids):
        ix=np.flatnonzero(half_ids==half)
        for a,b in known_runs(labels[ix]<3):
            a,b=a+ix[0],b+ix[0]
            if b-a<3:continue
            usable[a+1:b-1]=True
            for p in np.flatnonzero(labels[a+1:b]!=labels[a:b-1])+a+1:y[max(a,p-1):min(b,p+2)]=1
    return y,usable


def external_scores(data,probability):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
    import possession_metrics as pm
    prediction=np.full(len(data['full_labels']),3,np.int8)
    for half,first,last in data['metadata']['halves']:
        ix=data['half_ids']==half;frames=np.arange(first,last+1)
        p=np.stack([np.interp(frames,data['frames'][ix],probability[ix,c]) for c in range(3)],axis=1)
        prediction[first:last+1]=p.argmax(1)
    score=pm.evaluate(data['full_labels'],prediction,bootstrap_n=0)
    return {k:score[k] for k in ('spell_f1_mean','macro_f1','accuracy','confusion','n_gt_spells','n_pred_spells')},prediction


def train(args):
    out=Path(args.out)
    if out.exists():raise ValueError('Use a fresh external dead training directory')
    out.mkdir(parents=True);deterministic(0);rng=np.random.default_rng(0);source=[];hashes={}
    for game in (1,2):
        path=Path(args.features)/f'game_{game}.npz';hashes[str(game)]=sha(path)
        with np.load(path,allow_pickle=False) as d:
            data={k:d[k] for k in ('x','point','masked_x','masked_point','labels','half_ids')};meta=json.loads(str(d['metadata']))
        if meta['held_out'] or meta['target_labels_used']:raise ValueError('Wrong source for external dead fitting')
        data['half_ids']=data['half_ids']+game*10;source.append(data)
    data={k:np.concatenate([s[k] for s in source]) for k in source[0]};selected=data['labels']<3
    fit=baseline_fit(data['x'],data['point'],data['labels']);np.savez(out/'baseline.npz',**fit)
    mean=data['x'].mean((0,1));scale=np.maximum(data['x'].std((0,1)),1e-3)
    normalized={k:np.clip((data[k]-mean)/scale,-8,8).astype(np.float32) for k in ('x','masked_x')}
    np.savez(out/'normalization.npz',mean=mean,scale=scale)
    y,bmask=boundary_targets(data['labels'],data['half_ids']);positive=float(y[bmask].sum());negative=int(bmask.sum())-positive
    if min(positive,negative)<=0:raise ValueError('Need both external boundary classes')
    pos_weight=min(20.,negative/positive);model=DeadControlNet().to(args.device)
    live=int((data['labels']<2).sum());dead=int((data['labels']==2).sum())
    nn.init.constant_(model.dead.bias,np.log(dead/live));nn.init.constant_(model.boundary.bias,np.log(pos_weight*positive/negative))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60)
    halves=[(int(h),int(np.flatnonzero(data['half_ids']==h)[0]),int(np.flatnonzero(data['half_ids']==h)[-1])) for h in np.unique(data['half_ids'])]
    plan=dict(kind='external_temporal_dead_control',epochs=60,seed=0,core=256,batch=8,learning_rate=.001,weight_decay=.001,
        boundary_weight=.25,boundary_pos_weight=pos_weight,training_games=[1,2],external_test_game=3,target_labels_used=False,
        training_feature_sha256=hashes,protocol_sha256=sha(Path(__file__).resolve().parents[1]/'F1_TARGET.md'),
        parameters=sum(p.numel() for p in model.parameters()),early_stopping=False,external_comparison_predeclared_before_game3_metrics=True)
    save_json(out/'experiment.json',plan);logs=[]
    for epoch in range(60):
        masked=random_blocks(data['half_ids'],rng)
        x=np.where(masked[:,None,None],normalized['masked_x'],normalized['x'])
        point=np.where(masked[:,None],data['masked_point'],data['point'])
        tiles=epoch_tiles(selected,halves,256,rng);batches=(len(tiles)+7)//8;total=0.;model.train()
        for offset in range(0,len(tiles),8):
            xx=[];pp=[];yy=[];mm=[];by=[];bm=[]
            for start,first,stop,a,b in tiles[offset:offset+8]:
                xx.append(block_context(x,start,256,30,first,stop).transpose(1,2,0));pp.append(block_context(point,start,256,30,first,stop).T)
                indices=np.arange(start,start+256);clipped=indices.clip(0,len(selected)-1);inside=(indices>=a)&(indices<b)
                yy.append(data['labels'][clipped]);mm.append(inside&selected[clipped]);by.append(y[clipped]);bm.append(inside&bmask[clipped])
            tensor=lambda v:torch.as_tensor(np.stack(v),device=args.device)
            logits,q=model(tensor(xx),tensor(pp));logits=logits[:,:,30:286];q=q[:,30:286]
            ce=F.cross_entropy(logits,tensor(yy).long().clamp_max(2),reduction='none')
            bl=F.binary_cross_entropy_with_logits(q,tensor(by),pos_weight=q.new_tensor(pos_weight),reduction='none')
            loss=(ce*tensor(mm)).sum()*batches/int(selected.sum())+.25*(bl*tensor(bm)).sum()*batches/int(bmask.sum())
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())/batches
        scheduler.step()
        if not np.isfinite(total):raise ValueError('Nonfinite external dead loss')
        logs.append(total);print('EXTERNAL DEAD epoch',epoch+1,'/60 loss',total,flush=True)
    torch.save(model.state_dict(),out/'control.pt');save_json(out/'fit.json',dict(loss=logs,model_sha256=sha(out/'control.pt')))
    # One fixed comparison, after both fits. No source-3 fit/selection/normalization.
    path=Path(args.features)/'game_3.npz'
    with np.load(path,allow_pickle=False) as d:
        test={k:d[k] for k in ('x','point','frames','half_ids','full_labels')};test['metadata']=json.loads(str(d['metadata']))
    if not test['metadata']['held_out'] or test['metadata']['target_labels_used']:raise ValueError('Wrong external dead transfer fixture')
    logits,q=infer(model,test['x'],test['point'],test['half_ids'],mean,scale,args.device)
    baseline,_=external_scores(test,baseline_probability(test['x'],test['point'],fit));trained,pred=external_scores(test,softmax(logits,axis=1))
    keys=('spell_f1_mean','macro_f1');passed=all(trained[k]>=baseline[k] for k in keys) and any(trained[k]>baseline[k] for k in keys)
    result=dict(baseline=baseline,temporal=trained,gate_passed=passed,held_out_feature_sha256=sha(path),
        model_sha256=sha(out/'control.pt'),target_labels_used=False,target_match_score=False,torch=torch.__version__,device=args.device,
        limitation='Known weak public-event spans; two predeclared external hypotheses, not an untouched final test or target-match score')
    np.savez_compressed(out/'held_out_predictions.npz',logits=logits,boundary=expit(q-np.log(pos_weight)),prediction=pred)
    save_json(out/'report.json',result);print('EXTERNAL DEAD COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu');train(parser.parse_args())
