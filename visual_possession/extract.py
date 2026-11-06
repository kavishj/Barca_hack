"""Cache frozen visual observations, independently of possession label values."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from temporal_possession.audio import sha
from temporal_possession.network import deterministic
from .geometry import ViewGeometry,crop_views
from .media import decode_clips,normalize_rgb
from .cache import VisualCache
from .internvideo import load,download


def selected_centers(geometry,intervals):
    selected=np.zeros(len(geometry.centers),bool)
    for first,stop in intervals:
        selected|=(geometry.centers>=first)&(geometry.centers<stop)
    return selected


def worker(args):
    deterministic(0)
    geometry=ViewGeometry.load(args.geometry)
    selection=json.loads(Path(args.selection).read_text())
    indices=np.array(selection['indices'],int)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    embeddings=np.zeros((len(indices),3,512),np.float32)
    availability=np.zeros((len(indices),3),np.float32)
    completed=0
    if out.exists():
        with np.load(out,allow_pickle=False) as d:
            if not np.array_equal(d['indices'],indices):raise ValueError('Feature worker output belongs to a different selection')
            completed=int(d['completed']);embeddings[:completed]=d['embeddings'][:completed]
            availability[:completed]=d['availability'][:completed]
    model,info=load(args.model_cache,devices=args.devices.split(','))
    started=time.monotonic();encoded=0
    lookup={int(geometry.centers[index]):(i,int(index)) for i,index in enumerate(indices) if i>=completed}
    for center,rgb in decode_clips(args.video,sorted(lookup),geometry.halves):
        row,index=lookup[center]
        cropped,valid=crop_views(rgb,geometry.boxes[index],geometry.valid[index])
        pictures=[rgb,*cropped];valid=np.r_[1.,valid]
        for view in (0,1,2):
            if valid[view]<=0:continue
            with torch.inference_mode():feature=model(normalize_rgb(pictures[view])[None]).cpu().numpy()[0]
            if not np.isfinite(feature).all() or not np.isclose(np.linalg.norm(feature),1.,atol=1e-4):
                raise ValueError('Frozen feature is non-finite or not unit-normalized')
            embeddings[row,view]=feature;availability[row,view]=valid[view];encoded+=1
        completed=row+1
        if completed%25==0 or completed==len(indices):
            info.update(encoded_clips_this_session=encoded,elapsed_seconds=time.monotonic()-started,
                        peak_gpu_bytes=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
            np.savez_compressed(out,indices=indices,embeddings=embeddings,availability=availability,
                                completed=np.array(completed),metadata=np.array(json.dumps(info)))
            print(f'Visual cache worker {args.devices}: {completed}/{len(indices)} centers, '
                  f'{encoded} clips, {time.monotonic()-started:.1f}s',flush=True)
    print('VISUAL FEATURE WORKER COMPLETE',args.devices,flush=True)


def extract(args):
    geometry=ViewGeometry.load(args.geometry)
    if sha(args.video)!=geometry.provenance['video_sha256']:
        raise ValueError('Video differs from the automatically prepared crop geometry')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    intervals=json.loads(Path(args.intervals).read_text())['intervals'] if args.intervals else [
        [first,last+1] for _,first,last in geometry.halves]
    selected=selected_centers(geometry,intervals)
    if not selected.any():raise ValueError('No video centers selected')
    plan=dict(geometry_sha256=sha(args.geometry),video_sha256=sha(args.video),intervals=intervals,
              centers=int(selected.sum()),full_match=bool(selected.all()),
              frame_offsets=[-18,-6,6,18],views=['wide','home_candidate','away_candidate'],
              possession_labels_read=False)
    (out/'extraction_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    download(args.model_cache)
    count=min(2,torch.cuda.device_count())
    if not count:raise ValueError('Frozen video extraction requires CUDA')
    indices=np.flatnonzero(selected)
    single_gpu=True
    if count==2:
        # Probe one complete frozen encoder on a T4 before starting two copies.
        probe=out/'probe_selection.json'
        probe.write_text(json.dumps(dict(indices=[int(indices[0])]))+'\n')
        result=subprocess.run([sys.executable,'-m','visual_possession','worker','--video',args.video,
            '--geometry',args.geometry,'--selection',str(probe),'--model-cache',args.model_cache,
            '--out',str(out/'single_gpu_probe.npz'),'--devices','cuda:0'],env=dict(os.environ,HF_HUB_OFFLINE='1'))
        single_gpu=result.returncode==0
        print('Single-T4 complete encoder probe:',single_gpu,flush=True)
    if not single_gpu:count=1
    groups=[x for x in np.array_split(indices,count) if len(x)]
    processes=[]
    for part,group in enumerate(groups):
        selection=out/f'selection_{part}.json'
        selection.write_text(json.dumps(dict(indices=group.tolist()))+'\n')
        output=out/f'part_{part}.npz'
        command=[sys.executable,'-m','visual_possession','worker','--video',args.video,
            '--geometry',args.geometry,'--selection',str(selection),'--model-cache',args.model_cache,
            '--out',str(output),'--devices',f'cuda:{part}' if single_gpu else 'cuda:0,cuda:1']
        env=dict(os.environ,HF_HUB_OFFLINE='1')
        processes.append(subprocess.Popen(command,env=env))
    failed=[]
    for part,process in enumerate(processes):
        code=process.wait()
        if code:failed.append((part,code))
    if failed:raise RuntimeError(f'Visual feature workers failed: {failed}')
    embeddings=np.zeros((len(geometry.centers),3,512),np.float32)
    availability=np.zeros((len(geometry.centers),3),np.float32)
    runtime=[]
    for part in range(len(groups)):
        with np.load(out/f'part_{part}.npz',allow_pickle=False) as d:
            if int(d['completed'])!=len(d['indices']):raise ValueError('Incomplete visual feature worker output')
            embeddings[d['indices']]=d['embeddings'];availability[d['indices']]=d['availability']
            runtime.append(json.loads(str(d['metadata'])))
    plan['workers']=runtime
    cache=VisualCache(geometry.centers,geometry.half_ids,embeddings,availability,geometry.halves,geometry.fps,plan)
    cache.save(out/'visual_cache.npz')
    print('FROZEN VISUAL CACHE COMPLETE',len(indices),'centers; full match',plan['full_match'],flush=True)
