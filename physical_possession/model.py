import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from structured_possession.features import event_history


def features(cache):
    index={n:i for i,n in enumerate(cache.names)};fields={}
    def col(n):return cache.x[:,index[n]]
    def put(n,x):fields[n]=np.asarray(x,np.float32)
    dh,da=col('d_home'),col('d_away')
    put('relative_distance',(da-dh)/(da+dh+1))
    put('near_score',np.exp(-dh/3)-np.exp(-da/3))
    put('uncontested',((dh<1.5)&(da>=2)&(col('ball_speed')<8)).astype(float)-
        ((da<1.5)&(dh>=2)&(col('ball_speed')<8)).astype(float))
    for suffix in ('within_2m','within_5m','speed_mean','moving_fraction','spread_x','spread_y',
                   'nearest_relative_speed','nearest_radial_speed'):
        put(suffix,col('home_'+suffix)-col('away_'+suffix))
    put('tag_phase',np.tanh(col('tag_phase')));put('tag_restart',col('tag_restart'))
    for seconds in (1,3,5):put('distance_mean_'+str(seconds),col('distance_margin_mean_'+str(seconds)+'s'))
    for kind in ('contact','reception'):
        owner=col(kind+'_last_team');age=col(kind+'_age_s')
        future=np.zeros(len(cache.x),np.float32);until=np.full(len(cache.x),np.nan,np.float32)
        for _,first,last in cache.halves:
            event=col(kind+'_team_now')[first:last+1]
            who,when=event_history(event[::-1],cache.fps)
            future[first:last+1]=who[::-1];until[first:last+1]=when[::-1]
        for seconds in (.5,2,8):
            put(f'{kind}_past_{seconds}',np.where(np.isfinite(age),owner*np.exp(-age/seconds),0))
            put(f'{kind}_future_{seconds}',np.where(np.isfinite(until),future*np.exp(-until/seconds),0))
    x=np.column_stack(list(fields.values()));x[~cache.graded_mask()]=0
    return np.nan_to_num(x,nan=0,posinf=0,neginf=0),tuple(fields)


def fit(x,codes,halves,step=5,threads=4):
    selected=np.zeros(len(codes),bool)
    for _,first,last in halves:selected[first:last+1:step]=True
    selected&=codes<2
    if set(codes[selected].tolist())!={0,1}:raise ValueError('Physical training requires both live teams')
    scale=np.sqrt(np.mean(x[selected].astype(np.float64)**2,axis=0));scale[scale<1e-5]=1
    with threadpool_limits(limits=threads):
        model=LogisticRegression(C=.1,fit_intercept=False,max_iter=2000,tol=1e-8,random_state=0).fit(x[selected]/scale,(codes[selected]==0).astype(np.int8))
    if model.n_iter_[0]>=2000:raise ValueError('Ownership solver did not converge')
    return dict(coef=model.coef_[0],scale=scale,training_frames=int(selected.sum()),iterations=int(model.n_iter_[0]))


def probability(model,x):return expit(x/model['scale']@model['coef'])


def mix(graph_probability,owner,weight):
    if not 0<=weight<=1:raise ValueError('Invalid physical mixture weight')
    p=np.asarray(graph_probability,np.float64);owner=np.asarray(owner)
    if p.ndim!=2 or p.shape!=(len(owner),3) or not np.isfinite(owner).all() or ((owner<0)|(owner>1)).any():
        raise ValueError('Invalid ownership posterior')
    live=1-p[:,2];old=np.divide(p[:,0],live,out=np.full(len(live),.5),where=live>1e-12)
    home=(1-weight)*old+weight*owner
    result=np.column_stack([live*home,live*(1-home),p[:,2]]).astype(np.float32)
    if not np.isfinite(result).all() or (result<0).any() or not np.allclose(result.sum(axis=1),1):
        raise ValueError('Invalid graph probabilities')
    return result
