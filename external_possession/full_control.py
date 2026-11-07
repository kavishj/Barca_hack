"""Weak control sequences from public Metrica event semantics only."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from relational_possession.data import build_graph
from structured_possession.features import derivative
from .data import COMMIT,load_tracking,pass_labels


def read_events(path):
    path=Path(path)
    if path.suffix=='.csv':return pd.read_csv(path)
    result=[]
    for e in json.loads(path.read_text())['data']:
        sub=e.get('subtypes');parts=sub if isinstance(sub,list) else [sub] if sub else []
        team=(e.get('team') or {}).get('id')
        result.append({'Team':{'FIFATMA':'Home','FIFATMB':'Away'}.get(team,''),
            'Type':e['type']['name'],'Subtype':'-'.join(p['name'] for p in parts),'Period':int(e['period']),
            'Start Frame':int((e.get('start') or {}).get('frame') or 0),
            'End Frame':int((e.get('end') or {}).get('frame') or 0)})
    return pd.DataFrame(result)


def control_labels(events,periods):
    actions=defaultdict(list);n=len(periods)
    for row in events.to_dict('records'):
        kind=str(row['Type']).upper();sub=str(row.get('Subtype','')).upper();team={'Home':0,'Away':1}.get(row['Team'])
        first,stop=int(row['Start Frame']),int(row['End Frame']);period=int(row['Period'])
        if not 0<=first<n or periods[first]!=period:continue
        end=stop if 0<=stop<n and stop>=first and periods[stop]==period else first
        if kind in ('PASS','RECOVERY','SET PIECE','SHOT') and team is not None and 'OFFSIDE' not in sub:
            actions[first].append(('control',team))
        if kind=='BALL LOST' and team is not None:actions[end].append(('loss',team))
        if kind=='SHOT' and any(t in sub.split('-') for t in ('WOODWORK','BLOCKED','SAVED')) and team is not None:
            actions[end].append(('loss',team))
        if kind=='BALL OUT' or kind=='SHOT' and any(t in sub.split('-') for t in ('GOAL','OUT')):
            actions[end].append(('dead',2))
        if kind in ('FAULT RECEIVED','FOUL RECEIVED'):actions[first].append(('dead',2))
        if 'OFFSIDE' in sub or kind=='BALL LOST' and 'END HALF' in sub:actions[end].append(('dead',2))
    codes=np.full(n,3,np.int8);blocked=np.zeros(n,bool);conflicts=0
    for half in sorted(set(periods.tolist())-{0}):
        indices=np.flatnonzero(periods==half);first,last=int(indices[0]),int(indices[-1])
        current=2;conflict_active=False;cursor=first;points=sorted(p for p in actions if first<=p<=last)
        for point in points:
            codes[cursor:point]=current;blocked[cursor:point]=conflict_active;items=actions[point]
            for kind,team in items:
                if kind=='loss' and current==team:current=3
            clubs={team for kind,team in items if kind=='control'}
            if len(clubs)==1:current=next(iter(clubs));conflict_active=False
            elif len(clubs)>1:current=3;conflict_active=True;conflicts+=1
            if any(kind=='dead' for kind,_ in items):current=2;conflict_active=False
            cursor=point
        # A known stop can persist; an unresolved live tail is censored.
        codes[cursor:last+1]=2 if current==2 else 3
        blocked[cursor:last+1]=conflict_active
    closed=pass_labels(events,periods);known_pass=closed<2
    contradict=known_pass&(codes!=3)&(codes!=closed)
    fill=known_pass&(codes==3)&~blocked;codes[fill]=closed[fill];codes[contradict]=3
    return codes,dict(conflicting_control_instants=conflicts,pass_fill_frames=int(fill.sum()),
        pass_conflict_frames=int(contradict.sum()),frame_counts={str(c):int((codes==c).sum()) for c in range(4)})


def load_fifa(metadata_path,tracking_path):
    root=ET.parse(metadata_path).getroot();fps=int(root.findtext('.//FrameRate'))
    if fps!=25:raise ValueError('Pinned FIFA source must be 25 Hz')
    parameters={p.findtext('Name'):p.findtext('Value') for p in root.findall('.//GlobalConfig/ProviderGlobalParameters/ProviderParameter')}
    halves=tuple((i,int(parameters[name+'_start']),int(parameters[name+'_end'])) for i,name in ((1,'first_half'),(2,'second_half')))
    n=halves[-1][2]+1;periods=np.zeros(n,np.int8)
    for half,first,last in halves:periods[first:last+1]=half
    slots={};used={0:0,1:24};keepers={}
    for p in root.findall('.//Players/Player'):
        club={'FIFATMA':0,'FIFATMB':1}.get(p.attrib['teamId'])
        if club is None:raise ValueError('Unknown external FIFA team')
        params={v.findtext('Name'):v.findtext('Value') for v in p.findall('./ProviderPlayerParameters/ProviderParameter')}
        if str(params.get('position_type','')).casefold()=='goalkeeper':
            slot=23 if club==0 else 45;keepers[p.attrib['id']]=p.findtext('Name')
        else:
            slot=used[club];used[club]+=1
            if slot>=(23 if club==0 else 45):raise ValueError('External roster exceeds padded team capacity')
        slots[p.attrib['id']]=slot
    channels={p.attrib['id']:(p.attrib['playerId'],p.attrib['channelId']) for p in root.findall('.//PlayerChannels/PlayerChannel')}
    specifications=[]
    for s in root.findall('./DataFormatSpecifications/DataFormatSpecification'):
        groups=s.findall('./SplitRegister')[0].findall('./SplitRegister');order=[]
        for group in groups:
            refs=group.findall('./PlayerChannelRef')
            if len(refs)!=2:raise ValueError('Expected paired FIFA player coordinates')
            x,y=(channels[r.attrib['playerChannelId']] for r in refs)
            if x[0]!=y[0] or x[1]!='x' or y[1]!='y':raise ValueError('FIFA coordinate channel order changed')
            order.append(slots[x[0]])
        if len(set(order))!=len(order):raise ValueError('Two active goalkeepers share a canonical slot')
        specifications.append((int(s.attrib['startFrame']),int(s.attrib['endFrame']),order))
    specifications.sort();players=np.full((n,46,2),np.nan,np.float32);ball=np.full((n,2),np.nan,np.float32)
    previous=0;index=0
    with Path(tracking_path).open() as handle:
        for line in handle:
            if not line.strip():continue
            parts=line.strip().split(':');frame=int(parts[0])
            if frame!=previous+1 or frame>=n:raise ValueError('FIFA frame IDs are not continuous')
            while frame>specifications[index][1]:index+=1
            first,last,order=specifications[index]
            if not first<=frame<=last or len(parts)!=3:raise ValueError('FIFA format section mismatch')
            xy=np.fromstring(parts[1].replace(';',','),sep=',',dtype=np.float32)
            b=np.fromstring(parts[2],sep=',',dtype=np.float32)
            if len(xy)!=2*len(order) or b.shape!=(2,):raise ValueError('FIFA observation count differs from metadata')
            players[frame,order]=xy.reshape(-1,2)*np.array([105.,68.],np.float32)
            ball[frame]=b*np.array([105.,68.],np.float32);previous=frame
    if previous!=n-1:raise ValueError('External FIFA track ended early')
    speed=np.full((n,46),np.nan,np.float32)
    for _,first,last in halves:speed[first:last+1]=np.linalg.norm(derivative(players[first:last+1],25),axis=2)
    speed[speed>15]=np.nan;roles=np.r_[np.zeros(24,np.int8),np.ones(22,np.int8)]
    match=SimpleNamespace(ball=ball,players=players,player_speed=speed,fps=25,pitch=(105.,68.),halves=halves,
        n_frames=n,team_by_frame=lambda:np.broadcast_to(roles,(n,46)))
    return match,periods,keepers


def prepare(args):
    root,out=Path(args.external_dir),Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifests=[json.loads(Path(p).read_text()) for p in (args.manifest,args.fifa_manifest)]
    for m in manifests:
        if m['commit']!=COMMIT:raise ValueError('Full-control external source revision changed')
        for name,info in m['files'].items():
            if name.endswith(('.csv','.xml','.txt','.json')) and hashlib.sha256((root/name).read_bytes()).hexdigest()!=info['sha256']:
                raise ValueError('Full-control external source hash changed')
    for game in (1,2,3):
        prefix=root/f'data/Sample_Game_{game}'/f'Sample_Game_{game}_'
        if game<3:
            match,periods,keepers=load_tracking(str(prefix)+'RawTrackingData_Home_Team.csv',str(prefix)+'RawTrackingData_Away_Team.csv')
            event_path=Path(str(prefix)+'RawEventsData.csv')
        else:
            match,periods,keepers=load_fifa(str(prefix)+'metadata.xml',str(prefix)+'tracking.txt');event_path=Path(str(prefix)+'events.json')
        codes,stats=control_labels(read_events(event_path),periods);graph=build_graph(match)
        metadata=dict(game=game,source_commit=COMMIT,halves=graph.halves,fps=25,coarse_hz=5,
            target_labels_used=False,supervision='Weak public event-derived control; D/X excluded from owner loss',
            keepers=keepers,label_statistics=stats,held_out=game==3)
        np.savez_compressed(out/f'game_{game}.npz',**{name:getattr(graph,name) for name in ('nodes','roles','valid','positioned','confidence','frames','half_ids')},
            labels=codes[graph.frames],full_labels=codes,metadata=np.array(json.dumps(metadata)))
        print('FULL CONTROL',game,stats,'coarse',len(graph.frames),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('external-dir','manifest','fifa-manifest','out'):parser.add_argument('--'+name,required=True)
    prepare(parser.parse_args())
