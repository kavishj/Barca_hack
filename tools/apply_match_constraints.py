"""Apply supplied match facts to an algorithm-generated prediction CSV.

No reference labels are read. The original prediction remains a separate file.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learned_possession.features import FeatureCache
from learned_possession.__main__ import digest,read_labels,write_labels
from structured_possession.features import known_dead


def apply(features,predictions,out):
    if Path(out).resolve()==Path(predictions).resolve():
        raise ValueError('Keep the original prediction file; choose a separate output')
    cache=FeatureCache.load(features)
    codes=read_labels(predictions,len(cache.x))
    mask=known_dead(cache)
    changed=int((codes[mask]!=2).sum())
    codes[mask]=2
    write_labels(out,np.array(['H','A','D','X'])[codes])
    audit=dict(feature_sha256=digest(features),prediction_sha256=digest(predictions),
               output_sha256=digest(out),known_dead_frames=int(mask.sum()),changed_frames=changed,
               source='README.md supplied pre-kickoff interval: 110808..113741 inclusive',
               reference_labels_read=False,manual_prediction_edits=False)
    Path(str(out)+'.audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','predictions','out'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    print(json.dumps(apply(args.features,args.predictions,args.out),indent=2))
