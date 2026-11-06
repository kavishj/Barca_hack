"""Automatic homography-based candidate-player crops; no possession labels."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .media import clip_frames


@dataclass
class ViewGeometry:
    centers:np.ndarray
    half_ids:np.ndarray
    boxes:np.ndarray
    valid:np.ndarray
    slots:np.ndarray
    anchor_age:np.ndarray
    halves:tuple
    fps:int
    provenance:dict

    def save(self,path):
        np.savez_compressed(path,centers=self.centers,half_ids=self.half_ids,boxes=self.boxes,
            valid=self.valid,slots=self.slots,anchor_age=self.anchor_age,
            metadata=np.array(json.dumps(dict(halves=self.halves,fps=self.fps,provenance=self.provenance))))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            meta=json.loads(str(d['metadata']))
            return cls(*(d[k] for k in ('centers','half_ids','boxes','valid','slots','anchor_age')),
                tuple(map(tuple,meta['halves'])),meta['fps'],meta['provenance'])


def plan_views(match,width,height,step=50,side=256,max_anchor_age=50):
    if step<1 or side<32:raise ValueError('Invalid visual sampling/crop settings')
    roles=match.team_by_frame()
    centers=[];halves=[];boxes=[];valid=[];slots=[];ages=[]
    for half,first,last in match.halves:
        seen=np.isfinite(match.ball[first:last+1]).all(axis=1)
        previous=np.maximum.accumulate(np.where(seen,np.arange(first,last+1),-1000000))
        for center in np.unique(np.r_[np.arange(first,last+1,step),last]):
            centers.append(center);halves.append(half)
            age=center-int(previous[center-first]);ages.append(age/match.fps)
            candidate=np.full(2,-1,np.int16)
            rectangles=np.zeros((2,4,4),np.int32);observed=np.zeros((2,4),bool)
            if age<=max_anchor_age:
                ball=match.ball[int(previous[center-first])]
                distance=np.linalg.norm(match.players[center]-ball,axis=1)
                for team in (0,1):
                    ok=(roles[center]==team)&np.isfinite(distance)&(distance<=15)
                    if not ok.any():continue
                    slot=int(np.argmin(np.where(ok,distance,np.inf)));candidate[team]=slot
                    for j,frame in enumerate(clip_frames(int(center),match.halves)):
                        position=match.players[frame,slot]
                        h=match.homography[frame]
                        if not np.isfinite(position).all() or not np.isfinite(h).all():continue
                        if np.linalg.norm(position-match.players[center,slot])>12:continue
                        normalized=np.r_[position/np.array(match.pitch),1.]
                        pixel=h@normalized
                        if not np.isfinite(pixel).all() or abs(pixel[2])<1e-8:continue
                        x,y=pixel[:2]/pixel[2]
                        if not (0<=x<width and 0<=y<height):continue
                        # The projected tracking point is at the player's feet.
                        left=int(round(x-side/2));top=int(round(y-side/2-side/8))
                        rectangles[team,j]=[left,top,left+side,top+side]
                        observed[team,j]=True
            boxes.append(rectangles);valid.append(observed);slots.append(candidate)
    return ViewGeometry(np.array(centers),np.array(halves,np.int8),np.array(boxes),np.array(valid),
        np.array(slots),np.array(ages,np.float32),tuple(map(tuple,match.halves)),match.fps,
        dict(step_frames=step,crop_side_pixels=side,max_anchor_age_frames=max_anchor_age,
             candidate_rule='Nearest observed player per team within 15m of current/recent ball; no labels',
             video_width=width,video_height=height))


def crop_views(rgb,boxes,valid):
    result=[];availability=[]
    for team in (0,1):
        images=[]
        side=int(boxes[team,np.flatnonzero(valid[team])[0],2]-boxes[team,np.flatnonzero(valid[team])[0],0]) if valid[team].any() else 256
        for j,picture in enumerate(rgb):
            patch=np.full((side,side,3),127,np.uint8)
            if valid[team,j]:
                left,top,right,bottom=map(int,boxes[team,j]);height,width=picture.shape[:2]
                a,b=max(0,left),max(0,top);c,d=min(width,right),min(height,bottom)
                if a<c and b<d:patch[b-top:d-top,a-left:c-left]=picture[b:d,a:c]
            images.append(patch)
        result.append(np.stack(images))
        availability.append(float(valid[team].mean()) if valid[team].sum()>=3 else 0.)
    return result,np.array(availability,np.float32)


def prepare(args):
    import av,games
    from temporal_possession.audio import sha
    games.DATA_DIR=str(Path(args.data_dir).resolve())
    from possession import data
    data.CACHE_DIR=str(Path(args.out).resolve().parent/'raw_cache')
    match=data.load_match()
    with av.open(str(args.video)) as container:
        stream=container.streams.video[0];width,height=stream.width,stream.height
    geometry=plan_views(match,width,height,step=args.step)
    geometry.provenance.update(video_sha256=sha(args.video))
    geometry.save(args.out)
    print('Automatic visual geometry:',len(geometry.centers),'centers; crop valid fractions',geometry.valid.mean(axis=(0,2)).tolist(),flush=True)
