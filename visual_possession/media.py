"""Exact PTS-indexed RGB clips; preprocessing matches the released video model."""
from fractions import Fraction
import numpy as np
import torch
from torch.nn import functional as F

OFFSETS=np.array([-18,-6,6,18])


def clip_frames(center,halves):
    blocks=[h for h in halves if h[1]<=center<=h[2]]
    if len(blocks)!=1:raise ValueError('Video clip center is outside one graded half')
    _,first,last=blocks[0]
    return np.clip(center+OFFSETS,first,last)


def decode_clips(path,centers,halves):
    """Decode sorted clips, seeking only across gaps; preserve original pixels."""
    import av
    centers=list(map(int,centers))
    if centers!=sorted(set(centers)):raise ValueError('Use distinct increasing clip centers')
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        if stream.average_rate!=25:raise ValueError('Expected the supplied 25 fps tactical video')
        tick=Fraction(1,25)/stream.time_base
        if tick.denominator!=1:raise ValueError('Non-integral video timestamp grid')
        tick=int(tick);origin=stream.start_time or 0
        cache={};iterator=None;last=-100000
        for center in centers:
            requested=clip_frames(center,halves)
            start,stop=int(requested.min()),int(requested.max())
            if iterator is None or start>last+100 or start<min(cache,default=start):
                container.seek(origin+start*tick,stream=stream,backward=True)
                iterator=container.decode(stream);cache={}
            wanted=set(map(int,requested))
            for frame in iterator:
                if frame.pts is None:continue
                delta=frame.pts-origin
                if delta%tick:raise ValueError('Decoded video frame is not on the submission index grid')
                index=delta//tick;last=index
                if index>=start and index<=stop:cache[index]=frame.to_ndarray(format='rgb24')
                if index>=stop:break
            if not wanted<=set(cache):raise ValueError(f'Video seek missed a requested frame near {center}')
            yield center,np.stack([cache[int(k)] for k in requested])
            cache={k:v for k,v in cache.items() if k>=stop-75}


def normalize_rgb(rgb):
    values=torch.from_numpy(np.ascontiguousarray(rgb)).permute(0,3,1,2).float()/255
    values=F.interpolate(values,size=(224,224),mode='bilinear',align_corners=False)
    return (values-values.new_tensor([.485,.456,.406])[None,:,None,None])/values.new_tensor([.229,.224,.225])[None,:,None,None]
