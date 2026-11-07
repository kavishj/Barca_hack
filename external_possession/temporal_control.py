"""Fixed external-only temporal control training and held-out transfer gate."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import expit
from sklearn.metrics import f1_score
from relational_possession.model import SpatialEncoder, Config
from temporal_possession.network import ResidualBlock, deterministic, known_runs, epoch_tiles
from temporal_possession.audio import sha
from temporal_possession.experiment import save_json
from .data import COMMIT


def ball_mask(data):
    """Remove every ball-derived channel; preserve observed player motion."""
    result={k:v.copy() for k,v in data.items()}
    result['nodes'][:,:,5:10]=0;result['nodes'][:,:,14]=0
    result['nodes'][:,-1,:10]=0;result['nodes'][:,-1,12:15]=0
    result['positioned'][:,-1]=False
    return result


def pooled_features(encoder,data,coef,device,masked=False):
    values=ball_mask(data) if masked else data;parts=[]
    with torch.inference_mode():
        for start in range(0,len(values['nodes']),512):
            batch=[torch.as_tensor(values[k][start:start+512],device=device) for k in ('nodes','roles','valid','positioned')]
            parts.append(encoder(*batch).cpu().numpy())
    pooled=np.concatenate(parts);logit=(pooled[:,0].astype(np.float64)-pooled[:,1])@coef
    point=np.stack([.5*logit,-.5*logit],axis=1).astype(np.float32)
    seen=values['positioned'][:,-1].astype(np.float32)
    confidence=np.zeros(len(seen),np.float32) if masked else data['confidence']
    shared=np.broadcast_to(np.stack([seen,confidence],axis=1)[:,None],(len(seen),2,2))
    return np.concatenate([pooled,point[:,:,None],shared],axis=2),point


def extract(args):
    out=Path(args.out)
    if out.exists():raise ValueError('Use a fresh temporal feature directory')
    out.mkdir(parents=True);deterministic(0)
    info=json.loads(Path(args.pretraining_info).read_text())
    if info['target_match_labels_used'] or sha(args.pretrained)!=info['encoder_sha256']:
        raise ValueError('Frozen external encoder provenance mismatch')
    encoder=SpatialEncoder(15,Config()).to(args.device)
    encoder.load_state_dict(torch.load(args.pretrained,map_location=args.device,weights_only=True));encoder.eval()
    with np.load(args.probe,allow_pickle=False) as d:coef=d['coef']
    for game in (1,2,3):
        path=Path(args.graphs)/f'game_{game}.npz'
        with np.load(path,allow_pickle=False) as d:
            meta=json.loads(str(d['metadata']))
            if meta['target_labels_used'] or meta['source_commit']!=COMMIT or meta['held_out']!=(game==3):
                raise ValueError('Wrong external fixture provenance')
            data={k:d[k] for k in ('nodes','roles','valid','positioned','confidence')}
            carry={k:d[k] for k in ('labels','full_labels','frames','half_ids')}
        x,point=pooled_features(encoder,data,coef,args.device)
        extra={}
        if game<3:
            extra['masked_x'],extra['masked_point']=pooled_features(encoder,data,coef,args.device,masked=True)
        meta.update(graph_sha256=sha(path),encoder_sha256=sha(args.pretrained),probe_sha256=sha(args.probe))
        np.savez_compressed(out/f'game_{game}.npz',x=x,point=point,**extra,**carry,metadata=np.array(json.dumps(meta)))
        print('TEMPORAL features',game,x.shape,flush=True)


class ControlNet(nn.Module):
    radius=30
    def __init__(self):
        super().__init__()
        self.input=nn.Conv1d(35,32,1)
        self.blocks=nn.Sequential(*(ResidualBlock(32,d,.15) for d in (1,2,4,8)))
        self.owner=nn.Conv1d(32,1,1);nn.init.zeros_(self.owner.weight);nn.init.zeros_(self.owner.bias)
        self.boundary=nn.Conv1d(64,1,1)

    def forward(self,x,point):
        # B,teams,channels,time; all owner parameters are shared between teams.
        b,teams,c,t=x.shape
        h=self.blocks(F.gelu(self.input(x.reshape(b*teams,c,t)))).reshape(b,teams,32,t)
        logits=self.owner(h.reshape(b*teams,32,t)).reshape(b,teams,t)+point
        invariant=torch.cat([h.mean(1),(h[:,0]-h[:,1]).abs()],dim=1)
        return logits,self.boundary(invariant)[:,0]


def random_blocks(half_ids,rng):
    """Renewal gaps and uniform 1--20 s blocks: expected coverage about 20%."""
    mask=np.zeros(len(half_ids),bool)
    for half in np.unique(half_ids):
        indices=np.flatnonzero(half_ids==half);p=int(indices[0]);stop=int(indices[-1])+1
        while p<stop:
            p+=int(rng.geometric(1/210.));end=min(stop,p+int(rng.integers(5,101)))
            mask[p:end]=True;p=end
    return mask


def boundary_targets(labels,half_ids):
    selected=labels<2;y=np.zeros(len(labels),np.float32);usable=np.zeros(len(labels),bool)
    for half in np.unique(half_ids):
        ix=np.flatnonzero(half_ids==half)
        for a,b in known_runs(selected[ix]):
            a,b=a+ix[0],b+ix[0]
            if b-a<3:continue
            usable[a+1:b-1]=True
            changes=np.flatnonzero(labels[a+1:b]!=labels[a:b-1])+a+1
            for p in changes:y[max(a,p-1):min(b,p+2)]=1
    return y,usable


def block_context(values,start,core,radius,first,stop):
    result=np.zeros((core+2*radius,*values.shape[1:]),np.float32)
    a,b=max(first,start-radius),min(stop,start+core+radius)
    if b>a:result[a-start+radius:b-start+radius]=values[a:b]
    return result


def infer(model,x,point,half_ids,mean,scale,device):
    x=np.clip((x-mean)/scale,-8,8).astype(np.float32)
    logits=np.zeros((len(point),getattr(model,'states',2)),np.float32)
    boundary=np.zeros(len(x),np.float32);model.eval()
    with torch.inference_mode():
        for half in np.unique(half_ids):
            ids=np.flatnonzero(half_ids==half);first,stop=int(ids[0]),int(ids[-1])+1
            for start in range(first,stop,1024):
                a=block_context(x,start,1024,30,first,stop).transpose(1,2,0)[None]
                p=block_context(point,start,1024,30,first,stop).T[None]
                owner,q=model(torch.from_numpy(a).to(device),torch.from_numpy(p).to(device));length=min(1024,stop-start)
                logits[start:start+length]=owner[0,:,30:30+length].T.cpu().numpy()
                boundary[start:start+length]=q[0,30:30+length].cpu().numpy()
    return logits,boundary


def external_scores(data,logits):
    # Conditional live metrics only. References never enter predictions.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
    import possession_metrics as pm
    full=data['full_labels'];prediction=np.full(len(full),3,np.int8)
    coarse=expit(logits[:,0]-logits[:,1])
    for half,first,last in data['metadata']['halves']:
        ix=data['half_ids']==half
        p=np.interp(np.arange(first,last+1),data['frames'][ix],coarse[ix])
        prediction[first:last+1]=np.where(p>=.5,0,1)
    selected=full<2;runs=known_runs(selected)
    transition=pm.transition_scores(pm.transitions(full,runs),pm.transitions(prediction,runs),tol=12,typed='pair')
    return dict(conditional_live_owner_macro_f1=float(f1_score(full[selected],prediction[selected],labels=[0,1],average='macro',zero_division=0)),
        typed_turnover_f1_05s=transition['f1'],turnover_precision=transition['precision'],turnover_recall=transition['recall'],
        n_live_frames=int(selected.sum()),n_gt_turnovers=transition['n_gt'],n_pred_turnovers=transition['n_pred'],
        conditional_metrics=True,target_match_score=False),prediction


def train(args):
    out=Path(args.out)
    if out.exists():raise ValueError('Use a fresh temporal training directory')
    out.mkdir(parents=True);deterministic(0);rng=np.random.default_rng(0)
    source=[];hashes={}
    for game in (1,2):
        path=Path(args.features)/f'game_{game}.npz';hashes[str(game)]=sha(path)
        with np.load(path,allow_pickle=False) as d:
            data={k:d[k] for k in ('x','point','masked_x','masked_point','labels','half_ids')}
            meta=json.loads(str(d['metadata']))
        if meta['held_out'] or meta['target_labels_used']:raise ValueError('Held-out data cannot be fitted')
        data['half_ids']=data['half_ids']+game*10;source.append(data)
    data={k:np.concatenate([s[k] for s in source]) for k in source[0]}
    # Shared moments across teams, fit only on training fixture observations.
    mean=data['x'].mean((0,1));scale=data['x'].std((0,1));scale=np.maximum(scale,1e-3)
    normalized={k:np.clip((data[k]-mean)/scale,-8,8).astype(np.float32) for k in ('x','masked_x')}
    y,bmask=boundary_targets(data['labels'],data['half_ids']);selected=data['labels']<2
    halves=[(int(h),int(np.flatnonzero(data['half_ids']==h)[0]),int(np.flatnonzero(data['half_ids']==h)[-1])) for h in np.unique(data['half_ids'])]
    positive=float(y[bmask].sum());negative=int(bmask.sum())-positive
    if min(positive,negative)<=0:raise ValueError('External boundaries need both classes')
    pos_weight=min(20.,negative/positive);model=ControlNet().to(args.device)
    nn.init.constant_(model.boundary.bias,np.log(pos_weight*positive/negative))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60)
    plan=dict(kind='external_temporal_control',seed=0,epochs=60,core=256,batch=8,learning_rate=.001,weight_decay=.001,
        boundary_weight=.25,boundary_pos_weight=pos_weight,training_games=[1,2],external_test_game=3,target_labels_used=False,
        training_feature_sha256=hashes,protocol_sha256=sha(Path(__file__).resolve().parents[1]/'F1_TARGET.md'),
        normalization_games=[1,2],early_stopping=False,parameters=sum(p.numel() for p in model.parameters()))
    save_json(out/'experiment.json',plan);np.savez(out/'normalization.npz',mean=mean,scale=scale)
    logs=[]
    for epoch in range(60):
        masked=random_blocks(data['half_ids'],rng)
        x=np.where(masked[:,None,None],normalized['masked_x'],normalized['x'])
        point=np.where(masked[:,None],data['masked_point'],data['point'])
        tiles=epoch_tiles(selected,halves,256,rng);batches=(len(tiles)+7)//8;loss_total=0.;model.train()
        for offset in range(0,len(tiles),8):
            xx=[];pp=[];yy=[];mm=[];by=[];bm=[]
            for start,first,stop,a,b in tiles[offset:offset+8]:
                xx.append(block_context(x,start,256,30,first,stop).transpose(1,2,0))
                pp.append(block_context(point,start,256,30,first,stop).T)
                indices=np.arange(start,start+256);clipped=indices.clip(0,len(selected)-1)
                inside=(indices>=a)&(indices<b)
                yy.append(data['labels'][clipped]);mm.append(inside&selected[clipped])
                by.append(y[clipped]);bm.append(inside&bmask[clipped])
            tensor=lambda z:torch.as_tensor(np.stack(z),device=args.device)
            logits,q=model(tensor(xx),tensor(pp));logits=logits[:,:,30:286];q=q[:,30:286]
            target=tensor(yy).long().clamp_max(1);mask=tensor(mm);boundary_mask=tensor(bm)
            ce=F.cross_entropy(logits,target,reduction='none')
            bl=F.binary_cross_entropy_with_logits(q,tensor(by),reduction='none',pos_weight=q.new_tensor(pos_weight))
            loss=(ce*mask).sum()*batches/int(selected.sum())+.25*(bl*boundary_mask).sum()*batches/int(bmask.sum())
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            loss_total+=float(loss.detach())/batches
        schedule.step()
        if not np.isfinite(loss_total):raise ValueError('Nonfinite external control loss')
        logs.append(loss_total);print('EXTERNAL CONTROL epoch',epoch+1,'/60 loss',loss_total,'masked',float(masked.mean()),flush=True)
    torch.save(model.state_dict(),out/'control.pt');save_json(out/'fit.json',dict(loss=logs,model_sha256=sha(out/'control.pt')))
    # Open Game 3 only after final fitting; one comparison, no checkpoint selection.
    path=Path(args.features)/'game_3.npz'
    with np.load(path,allow_pickle=False) as d:
        test={k:d[k] for k in ('x','point','frames','half_ids','full_labels')};test['metadata']=json.loads(str(d['metadata']))
    if not test['metadata']['held_out'] or test['metadata']['target_labels_used']:raise ValueError('Wrong external transfer fixture')
    logits,q=infer(model,test['x'],test['point'],test['half_ids'],mean,scale,args.device)
    baseline,_=external_scores(test,test['point']);trained,pred=external_scores(test,logits)
    keys=('conditional_live_owner_macro_f1','typed_turnover_f1_05s')
    passed=all(trained[k]>=baseline[k] for k in keys) and any(trained[k]>baseline[k] for k in keys)
    result=dict(baseline=baseline,temporal=trained,gate_passed=passed,held_out_feature_sha256=sha(path),
        model_sha256=sha(out/'control.pt'),target_labels_used=False,torch=torch.__version__,device=args.device,
        limitation='Weak event-derived conditional external metrics; not target-match spell F1')
    np.savez_compressed(out/'held_out_predictions.npz',logits=logits,boundary=expit(q-np.log(pos_weight)),prediction=pred)
    save_json(out/'report.json',result);print('EXTERNAL CONTROL COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('extract')
    for name in ('graphs','pretrained','pretraining-info','probe','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');p=sub.add_parser('train')
    for name in ('features','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');args=parser.parse_args()
    (extract if args.command=='extract' else train)(args)
