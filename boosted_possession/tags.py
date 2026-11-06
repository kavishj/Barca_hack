"""Keep supplied automatic event types separate; never consume reference labels."""
import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
from learned_possession.features import FeatureCache
from temporal_possession.audio import sha


def events(path,fps):
    result=[]
    for item in ET.parse(path).iter('instance'):
        tags={p.findtext('group'):p.findtext('text') for p in item.findall('label')}
        team=(tags.get('Team') or '').casefold()
        club='home' if 'barcelona' in team else 'away' if 'athletic' in team else None
        if club is None:continue
        code=item.findtext('code') or ''
        if code=='SET PIECES':code+=' '+(tags.get('Type') or '')
        name=re.sub('[^a-z0-9]+','_',code.casefold()).strip('_')
        first=round(float(item.findtext('start'))*fps);stop=round(float(item.findtext('end'))*fps)
        if stop>first:result.append((club,name,first,stop))
    return result


def enrich(rich,observed):
    groups=defaultdict(list)
    for club,name,first,stop in observed:groups[club,name].append((first,stop))
    vocabulary=sorted({name for _,name in groups});columns=[];names=[]
    for name in vocabulary:
        for club in ('home','away'):
            active=np.zeros(len(rich.x),np.float32)
            start_offset=np.full(len(rich.x),np.nan,np.float32);end_offset=start_offset.copy()
            for _,left,last in rich.halves:
                for first,stop in groups[club,name]:
                    # An interval must overlap this half to supply context here.
                    if not first<=last or stop<=left:continue
                    a,b=max(left,first),min(last+1,stop)
                    if b>a:active[a:b]+=1
                    for point,values in ((first,start_offset),(stop,end_offset)):
                        a,b=max(left,point-30*rich.fps),min(last+1,point+30*rich.fps+1)
                        if b<=a:continue
                        offset=(np.arange(a,b)-point)/rich.fps
                        old=values[a:b];choose=~np.isfinite(old)|(np.abs(offset)<np.abs(old))
                        old[choose]=offset[choose]
            columns.extend([active,start_offset,end_offset])
            names.extend([f'{club}_auto_{name}_{suffix}' for suffix in ('active','start_offset_s','end_offset_s')])
    if not columns:raise ValueError('No team-tagged automatic observations')
    return FeatureCache(np.column_stack([rich.x,*columns]),(*rich.names,*names),rich.fps,rich.halves,
        dict(rich.provenance,automatic_tag_types=vocabulary,automatic_tag_context_seconds=30,no_reference_labels=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('rich-features','xml','out'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();rich=FeatureCache.load(args.rich_features)
    result=enrich(rich,events(args.xml,rich.fps));result.provenance['smart_tagging_sha256']=sha(args.xml)
    result.provenance['source_rich_sha256']=sha(args.rich_features);result.save(args.out)
    print('Automatic tag cache',len(result.x),'frames x',len(result.names),'features')
