"""Training-only capped live-spell weights; censored edges retain unit weight."""
import numpy as np


def spell_weights(codes):
    codes=np.asarray(codes)
    boundaries=np.r_[0,np.flatnonzero(codes[1:]!=codes[:-1])+1,len(codes)]
    complete=[]
    for first,stop in zip(boundaries[:-1],boundaries[1:]):
        if codes[first]<2 and first>0 and stop<len(codes) and codes[first-1]!=3 and codes[stop]!=3:
            complete.append((int(first),int(stop)))
    result=np.ones(len(codes),np.float32)
    complete_mask=np.zeros(len(codes),bool)
    median=float(np.median([b-a for a,b in complete])) if complete else 1.
    for first,stop in complete:
        result[first:stop]=np.clip(np.sqrt(median/(stop-first)),.25,4.)
        complete_mask[first:stop]=True
    if complete_mask.any():result[complete_mask]*=int(complete_mask.sum())/float(result[complete_mask].sum())
    result[codes==3]=0
    return result,dict(complete_live_spells=len(complete),median_complete_frames=median,
                      policy='capped inverse sqrt duration; censored unit before live-mass normalization')
