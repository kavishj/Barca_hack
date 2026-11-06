"""Observation-derived contact/reception history on the original graph nodes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from learned_possession.features import FeatureCache
from relational_possession.data import GraphCache
from temporal_possession.audio import sha

HISTORY_NAMES=('contact_decay_1s','contact_decay_3s','contact_decay_10s','contact_observed','contact_age',
    'reception_decay_1s','reception_decay_5s','reception_observed','reception_age',
    'uncontested_control_1s','uncontested_control_3s','ball_observed_1s')


def enrich(graph,cache):
    if graph.halves!=cache.halves or graph.fps!=cache.fps:raise ValueError('History/graph timelines differ')
    columns={name:i for i,name in enumerate(cache.names)}
    for name in ('contact_last_team','contact_age_s','reception_last_team','reception_age_s','d_home','d_away','ball_speed','ball_visible'):
        if name not in columns:raise ValueError('Missing observation feature '+name)
    frames=graph.frames;roles=graph.roles
    out=np.zeros((len(frames),47,len(HISTORY_NAMES)),np.float32)
    start=0
    for kind,taus in (('contact',(1.,3.,10.)),('reception',(1.,5.))):
        owner=cache.x[frames,columns[kind+'_last_team']]
        age=cache.x[frames,columns[kind+'_age_s']]
        known=np.isfinite(age)&(owner!=0)&(age<=30)
        membership=known[:,None]&(((owner[:,None]>0)&(roles==0))|((owner[:,None]<0)&(roles==1)))
        for tau in taus:
            out[:,:,start]=membership*np.exp(-np.nan_to_num(age,nan=100000)[:,None]/tau);start+=1
        out[:,:,start]=known[:,None];start+=1
        out[:,:,start]=np.where(known,np.log1p(np.nan_to_num(age))/np.log(31),0)[:,None];start+=1
    control=np.zeros((len(cache.x),2,2),np.float32);availability=np.zeros(len(cache.x),np.float32)
    for _,first,last in cache.halves:
        visible=cache.x[first:last+1,columns['ball_visible']]>0
        speed=cache.x[first:last+1,columns['ball_speed']]
        distances=[cache.x[first:last+1,columns[name]] for name in ('d_home','d_away')]
        availability[first:last+1]=pd.Series(visible.astype(float)).rolling(cache.fps,min_periods=1).mean()
        for team in (0,1):
            controlled=visible&(distances[team]<1.5)&(distances[1-team]>=2)&(speed<8)
            for j,seconds in enumerate((1,3)):
                control[first:last+1,team,j]=pd.Series(controlled.astype(float)).rolling(seconds*cache.fps,min_periods=1).mean()
    for team in (0,1):out[:,:,-3:-1]+=control[frames,team,None,:]*(roles==team)[:,:,None]
    out[:,:,-1]=availability[frames,None]
    if not np.isfinite(out).all():raise ValueError('Non-finite observation history')
    return GraphCache(graph.frames,graph.half_ids,np.concatenate([graph.nodes,out],axis=2),graph.roles,
        graph.valid,graph.positioned,graph.confidence,graph.halves,graph.fps)


def prepare(args):
    graph=GraphCache.load(args.graph);rich=FeatureCache.load(args.rich_features)
    if graph.nodes.shape[2]!=15:raise ValueError('Expected the original 15-feature graph')
    result=enrich(graph,rich)
    payload={key:getattr(result,key) for key in ('frames','half_ids','nodes','roles','valid','positioned','confidence')}
    payload['metadata']=np.array(json.dumps(dict(halves=result.halves,fps=result.fps,history_features=HISTORY_NAMES,
        base_graph_sha256=sha(args.graph),rich_features_sha256=sha(args.rich_features),labels_read=False)))
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,**payload)
    print('Contact history graph:',len(result.frames),'x 47 x',result.nodes.shape[2],flush=True)
