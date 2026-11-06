"""Pinned Metrica sample tracking and closed-pass weak supervision."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
import pandas as pd
from relational_possession.data import build_graph
from structured_possession.features import derivative

COMMIT='e706dd506b360d69d9d123d5b8026e7294b13996'


def pass_labels(events,periods):
    counts=np.zeros((len(periods),2),np.int16)
    for row in events.to_dict('records'):
        if row['Type']!='PASS' or 'OFFSIDE' in str(row.get('Subtype','')).upper():continue
        if row['Team'] not in ('Home','Away'):continue
        first,stop=int(row['Start Frame']),int(row['End Frame'])
        if not 0<=first<stop<len(periods):continue
        if not np.all(periods[first:stop]==int(row['Period'])):continue
        counts[first:stop,0 if row['Team']=='Home' else 1]+=1
    labels=np.full(len(periods),3,np.int8)
    labels[(counts[:,0]>0)&(counts[:,1]==0)]=0
    labels[(counts[:,1]>0)&(counts[:,0]==0)]=1
    return labels


def load_tracking(home_path,away_path):
    home,away=pd.read_csv(home_path,skiprows=2),pd.read_csv(away_path,skiprows=2)
    for name in ('Frame','Period','Time [s]'):
        if not np.array_equal(home[name],away[name]):raise ValueError('External team timelines differ')
    frames=home.Frame.to_numpy(int);n=int(frames[-1])+1
    if not np.array_equal(frames,np.arange(1,n)):raise ValueError('External frame IDs are not continuous 1-based IDs')
    periods=np.zeros(n,np.int8);periods[frames]=home.Period.to_numpy(np.int8)
    halves=[]
    for half in (1,2):
        indices=np.flatnonzero(periods==half)
        if len(indices):
            if not np.all(np.diff(indices)==1):raise ValueError('External period is not contiguous')
            halves.append((half,int(indices[0]),int(indices[-1])))
    players=np.full((n,46,2),np.nan,np.float32);keepers={}
    role=np.r_[np.zeros(24,np.int8),np.ones(22,np.int8)]
    for club,table,slots,gk in (('home',home,list(range(23)),23),('away',away,list(range(24,45)),45)):
        columns=[i for i,name in enumerate(table.columns) if str(name).startswith('Player')]
        if not columns or len(columns)>len(slots)+1:raise ValueError('External player schema unsupported')
        coordinates=np.stack([table.iloc[:,i:i+2].to_numpy(np.float32) for i in columns],axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            distance=np.nanmean(np.abs(coordinates[:,:,0]-.5),axis=0)
        keeper=int(np.nanargmax(distance));keepers[club]=str(table.columns[columns[keeper]])
        for k in range(len(columns)):
            slot=gk if k==keeper else slots.pop(0)
            players[frames,slot]=coordinates[:,k]*np.array([105.,68.],np.float32)
    hb,ab=home.iloc[:,-2:].to_numpy(np.float32),away.iloc[:,-2:].to_numpy(np.float32)
    if not np.allclose(hb,ab,equal_nan=True):raise ValueError('External copies of the ball track differ')
    ball=np.full((n,2),np.nan,np.float32);ball[frames]=hb*np.array([105.,68.],np.float32)
    speed=np.full((n,46),np.nan,np.float32)
    for _,first,last in halves:speed[first:last+1]=np.linalg.norm(derivative(players[first:last+1],25),axis=2)
    speed[speed>15]=np.nan
    match=SimpleNamespace(ball=ball,players=players,player_speed=speed,fps=25,pitch=(105.,68.),
        halves=tuple(halves),n_frames=n,team_by_frame=lambda:np.broadcast_to(role,(n,46)))
    return match,periods,keepers


def prepare(args):
    root=Path(args.external_dir);manifest=json.loads(Path(args.manifest).read_text())
    if manifest['commit']!=COMMIT:raise ValueError('External source revision changed')
    arrays={name:[] for name in ('nodes','roles','valid','positioned','labels','frames','games')};reports=[]
    for game,count in ((1,145006),(2,141156)):
        prefix=f'data/Sample_Game_{game}/Sample_Game_{game}_'
        paths={kind:root/(prefix+kind+'.csv') for kind in ('RawTrackingData_Home_Team','RawTrackingData_Away_Team','RawEventsData')}
        for path in paths.values():
            relative=str(path.relative_to(root));digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=manifest['files'][relative]['sha256']:raise ValueError('External source hash changed')
        match,periods,keepers=load_tracking(paths['RawTrackingData_Home_Team'],paths['RawTrackingData_Away_Team'])
        if match.n_frames!=count+1:raise ValueError('Wrong external fixture timeline')
        codes=pass_labels(pd.read_csv(paths['RawEventsData']),periods)
        graph=build_graph(match);selected=codes[graph.frames]<2
        for name in ('nodes','roles','valid','positioned'):arrays[name].append(getattr(graph,name)[selected])
        arrays['labels'].append(codes[graph.frames[selected]]);arrays['frames'].append(graph.frames[selected])
        arrays['games'].append(np.full(int(selected.sum()),game,np.int8))
        reports.append(dict(game=game,tracking_frames=count,closed_pass_frames=int((codes<2).sum()),
            supervised_graphs=int(selected.sum()),home_graphs=int((codes[graph.frames[selected]]==0).sum()),
            observed_ball_fraction=float(graph.positioned[selected,-1].mean()),keeper_slots=keepers))
        print('External fixture prepared',reports[-1],flush=True)
    metadata=dict(source='Metrica Sports sample data; acknowledgement in EXTERNAL_PRETRAINING.md',
        source_commit=COMMIT,source_manifest=manifest,fixtures=reports,supervision='Closed PASS only; X gaps; no D labels',
        target_match_labels_used=False)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,**{name:np.concatenate(parts) for name,parts in arrays.items()},metadata=np.array(json.dumps(metadata)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('external-dir','manifest','out'):parser.add_argument('--'+name,required=True)
    prepare(parser.parse_args())
