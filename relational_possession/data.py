"""Observation-only player/ball graph cache and team-shared preprocessing."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from temporal_possession.network import Normalizer
from structured_possession.features import derivative


@dataclass
class GraphCache:
    frames:np.ndarray
    half_ids:np.ndarray
    nodes:np.ndarray
    roles:np.ndarray
    valid:np.ndarray
    positioned:np.ndarray
    confidence:np.ndarray
    halves:tuple
    fps:int

    def __post_init__(self):
        self.halves=tuple(map(tuple,self.halves))
        n=len(self.frames)
        if (self.nodes.ndim!=3 or self.nodes.shape[:2]!=(n,47) or self.roles.shape!=(n,47)
                or self.valid.shape!=(n,47) or self.positioned.shape!=(n,47)
                or self.confidence.shape!=(n,) or self.half_ids.shape!=(n,)
                or not np.isfinite(self.nodes).all() or not np.isfinite(self.confidence).all()
                or np.any((self.confidence<0)|(self.confidence>1))):
            raise ValueError('Malformed player/ball graph cache')
        for half,first,last in self.halves:
            locations=self.frames[self.half_ids==half]
            if not len(locations) or locations[0]!=first or locations[-1]!=last or np.any(np.diff(locations)<=0):
                raise ValueError('Graph samples must cover each half, in order')
        if not np.all(self.valid[:,-1]) or not np.all(self.roles[:,-1]==2):
            raise ValueError('The final node must be the possibly unobserved ball token')

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,frames=self.frames,half_ids=self.half_ids,nodes=self.nodes,
                            roles=self.roles,valid=self.valid,positioned=self.positioned,
                            confidence=self.confidence,metadata=np.array(json.dumps(dict(halves=self.halves,fps=self.fps))))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            return cls(*(data[name] for name in ('frames','half_ids','nodes','roles','valid','positioned','confidence')),
                       tuple(map(tuple,meta['halves'])),meta['fps'])

    def window(self,first,stop,half):
        block=[h for h in self.halves if h[0]==half]
        if len(block)!=1 or not block[0][1]<=first<stop<=block[0][2]+1:
            raise ValueError('A graph window must remain inside one half')
        locations=np.flatnonzero(self.half_ids==half)
        frame=self.frames[locations]
        left=max(0,int(np.searchsorted(frame,first,side='right'))-1)
        right=min(len(frame)-1,int(np.searchsorted(frame,stop-1,side='left')))
        indices=locations[left:right+1]
        sampled=self.frames[indices]
        fine=np.arange(first,stop)
        before=np.clip(np.searchsorted(sampled,fine,side='right')-1,0,len(sampled)-1)
        after=np.minimum(before+1,len(sampled)-1)
        weight=(fine-sampled[before])/np.maximum(sampled[after]-sampled[before],1)
        return dict(nodes=self.nodes[indices],roles=self.roles[indices],valid=self.valid[indices],
                    positioned=self.positioned[indices],confidence=self.confidence[indices],
                    left=before,right=after,weight=weight.astype(np.float32),
                    coarse_to_fine=np.clip(sampled-first,0,stop-first-1))


def build_graph(match,step=5):
    if step<1:
        raise ValueError('Positive graph sampling step required')
    frames,ids,nodes,roles,valid,positioned,confidence=[],[],[],[],[],[],[]
    teams=match.team_by_frame()
    for half,first,last in match.halves:
        stop=last+1
        ball,players=match.ball[first:stop],match.players[first:stop]
        pv,bv=derivative(players,match.fps),derivative(ball,match.fps)
        pv[np.linalg.norm(pv,axis=2)>15]=np.nan
        bv[np.linalg.norm(bv,axis=1)>60]=np.nan
        indices=np.unique(np.r_[np.arange(0,stop-first,step),stop-first-1])
        for start in range(0,len(indices),512):
            ix=indices[start:start+512]
            ball_pos,player_pos=ball[ix],players[ix]
            pos=np.concatenate([player_pos,ball_pos[:,None]],axis=1)
            velocity=np.concatenate([pv[ix],bv[ix,None]],axis=1)
            seen=np.isfinite(pos).all(axis=2)
            velocity_seen=np.isfinite(velocity).all(axis=2)
            ball_seen=seen[:,-1]
            delta=pos-ball_pos[:,None]
            relative_velocity=velocity-bv[ix,None]
            speed=np.concatenate([match.player_speed[first+ix],np.linalg.norm(bv[ix],axis=1)[:,None]],axis=1)
            is_ball=np.zeros(seen.shape,np.float32);is_ball[:,-1]=1
            keeper=np.zeros(seen.shape,np.float32);keeper[:,[23,45]]=1
            values=np.concatenate([
                pos/np.array([match.pitch[0],match.pitch[1]])-.5,velocity/10,speed[:,:,None]/10,
                delta/20,np.linalg.norm(delta,axis=2)[:,:,None]/20,relative_velocity/10,
                is_ball[:,:,None],keeper[:,:,None],seen[:,:,None],velocity_seen[:,:,None],
                np.broadcast_to(ball_seen[:,None,None],(*seen.shape,1))],axis=2)
            values=np.clip(np.nan_to_num(values,nan=0,posinf=0,neginf=0),-8,8).astype(np.float32)
            available=seen.copy();available[:,-1]=True
            club=np.concatenate([teams[first+ix],np.full((len(ix),1),2,np.int8)],axis=1)
            distance=np.linalg.norm(player_pos-ball_pos[:,None],axis=2)
            nearest=np.min(np.where(np.isfinite(distance),distance,np.inf),axis=1)
            certainty=np.where(ball_seen,np.exp(-nearest/3),0).astype(np.float32)
            frames.extend((first+ix).tolist());ids.extend([half]*len(ix))
            nodes.append(values);roles.append(club);valid.append(available)
            positioned.append(seen);confidence.append(certainty)
    return GraphCache(np.array(frames),np.array(ids,np.int8),np.concatenate(nodes),np.concatenate(roles),
                       np.concatenate(valid),np.concatenate(positioned),np.concatenate(confidence),
                       tuple(map(tuple,match.halves)),match.fps)


def prepare(args):
    import games
    games.DATA_DIR=str(Path(args.data_dir).resolve())
    from possession import data
    data.CACHE_DIR=str(Path(args.out).resolve().parent/'raw_cache')
    graph=build_graph(data.load_match())
    graph.save(args.out)
    print(f'Graph cache: {len(graph.frames)} samples x 47 tokens x {graph.nodes.shape[-1]} features',flush=True)


@dataclass
class SymmetricNormalizer:
    audio_normalizer:Normalizer
    global_indices:np.ndarray
    pair_indices:np.ndarray
    pair_signs:np.ndarray
    global_mean:np.ndarray
    global_scale:np.ndarray
    pair_mean:np.ndarray
    pair_scale:np.ndarray

    @classmethod
    def fit(cls,cache,audio,selected,components=32):
        normalizer=Normalizer.fit(cache,audio,selected,components)
        names=list(normalizer.names)
        pair=[];signs=[];used=set()
        for i,name in enumerate(names):
            if name.startswith('home_'):
                other='away_'+name[5:]
                if other not in names:
                    raise ValueError('Home/away feature schemas must match')
                j=names.index(other);pair.append([i,j]);signs.append([1,1]);used.update([i,j])
        for a,b in [('d_home','d_away')]:
            if a in names and b in names:
                i,j=names.index(a),names.index(b)
                pair.append([i,j]);signs.append([1,1]);used.update([i,j])
        signed={'distance_margin','tag_restart','tag_phase','ball_x_home_direction',
                'ball_vx_home_direction','camera_x_home_direction'}
        signed.update(name for name in names if name.startswith('distance_margin_mean_'))
        for name in sorted(signed&set(names)):
            i=names.index(name);pair.append([i,i]);signs.append([1,-1]);used.add(i)
        global_indices=np.array([i for i in range(len(names)) if i not in used],int)
        pair_indices=np.array(pair,int).reshape(-1,2)
        pair_signs=np.array(signs,np.float32).reshape(-1,2)
        result=cls(normalizer,global_indices,pair_indices,pair_signs,None,None,None,None)
        g,p=result.raw(cache,audio)
        def stats(values):
            finite=np.isfinite(values)
            mean=np.divide(np.where(finite,values,0).sum(axis=0,dtype=np.float64),finite.sum(axis=0),
                           out=np.zeros(values.shape[-1]),where=finite.sum(axis=0)>0)
            var=np.divide(np.where(finite,(values-mean)**2,0).sum(axis=0,dtype=np.float64),finite.sum(axis=0),
                          out=np.ones(values.shape[-1]),where=finite.sum(axis=0)>0)
            return mean.astype(np.float32),np.maximum(np.sqrt(var),1e-5).astype(np.float32)
        result.global_mean,result.global_scale=stats(g[selected])
        values=p[selected].reshape(-1,p.shape[-1]) if p.shape[-1] else np.empty((2*int(selected.sum()),0))
        result.pair_mean,result.pair_scale=stats(values)
        return result

    def raw(self,cache,audio):
        raw=self.audio_normalizer.raw(cache,audio)
        return raw[:,self.global_indices],(raw[:,self.pair_indices]*self.pair_signs[None]).transpose(0,2,1)

    def transform(self,cache,audio):
        g,p=self.raw(cache,audio)
        def scale(values,mean,std):
            observed=np.isfinite(values)
            z=np.where(observed,(values-mean)/std,0)
            return np.concatenate([np.clip(z,-8,8),~observed],axis=-1).astype(np.float32)
        return scale(g,self.global_mean,self.global_scale),scale(p,self.pair_mean,self.pair_scale)

    def save(self,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        self.audio_normalizer.save(folder/'audio_normalizer.npz')
        np.savez_compressed(folder/'symmetric.npz',**{name:getattr(self,name) for name in
            ('global_indices','pair_indices','pair_signs','global_mean','global_scale','pair_mean','pair_scale')})

    @classmethod
    def load(cls,folder):
        folder=Path(folder)
        with np.load(folder/'symmetric.npz',allow_pickle=False) as values:
            return cls(Normalizer.load(folder/'audio_normalizer.npz'),*(values[name] for name in
                ('global_indices','pair_indices','pair_signs','global_mean','global_scale','pair_mean','pair_scale')))
