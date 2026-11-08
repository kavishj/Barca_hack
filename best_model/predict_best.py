"""Reproduce the preserved 0.7200 graph/audio candidate without reference labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
CODE=HERE if (HERE/'relational_possession').is_dir() else HERE.parent
sys.path.insert(0,str(CODE))


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def raw_paths(root,manifest):
    result={}
    for kind,info in manifest['raw_inputs'].items():
        found=list(Path(root).rglob(info['name']))
        if len(found)!=1:raise ValueError(f'Expected one {info["name"]} below {root}, found {len(found)}')
        if sha(found[0])!=info['sha256']:raise ValueError(f'Original input changed: {info["name"]}')
        result[kind]=str(found[0].resolve())
    return result


def prepare_raw(args,manifest):
    paths=raw_paths(args.data_dir,manifest)
    work=Path(args.work_dir or Path(args.out).resolve().parent/'.barca_inference_work');work.mkdir(parents=True,exist_ok=True)
    import games
    def asset_paths(game=games.GAME,data_dir=None,source='tactical'):
        if game!=games.GAME or source!='tactical':raise ValueError('This saved model is for BAR-ATH tactical only')
        return dict(paths)
    games.asset_paths=asset_paths
    from possession import data,audio,signals
    data.CACHE_DIR=audio.CACHE_DIR=str(work/'raw_cache')
    from learned_possession.features import build
    from relational_possession.data import build_graph
    from temporal_possession.audio import extract
    print('Preparing tracking and heuristic audio features from original assets.',flush=True)
    match=data.load_match();base=build(match,signals.compute(match),dict(raw_input_sha256={k:v['sha256'] for k,v in manifest['raw_inputs'].items()}))
    out=work/'observations';out.mkdir(exist_ok=True);base.save(out/'features.npz')
    build_graph(match).save(out/'graph.npz')
    print('Preparing frozen PANNs audio embeddings.',flush=True)
    extract(paths['video'],HERE/'model/panns/Cnn6_mAP=0.343.pth',out/'audio_cache.npz',match.halves,
        match.n_frames,match.fps,args.device,batch_size=32,wave_path=str(work/'panns_audio_32000.f32'))
    return out


def run(args):
    manifest=json.loads((HERE/'manifest.json').read_text())
    for name,expected in manifest.get('source_files_sha256',{}).items():
        if sha(HERE/name)!=expected:raise ValueError('Inference source changed: '+name)
    for name,expected in manifest['model_files_sha256'].items():
        if sha(HERE/name)!=expected:raise ValueError('Model/preprocessing file changed: '+name)
    from temporal_possession.network import deterministic
    deterministic(0)
    if args.data_dir:cache=prepare_raw(args,manifest)
    else:
        cache=Path(args.cache_dir or HERE/'data_cache')
        for name,expected in manifest['cache_files_sha256'].items():
            path=cache/name
            if not path.is_file():raise ValueError('Missing observation cache; use --data-dir or the ready-to-run bundle: '+str(path))
            if sha(path)!=expected:raise ValueError('Saved observation cache differs: '+name)
    from learned_possession.features import FeatureCache
    from temporal_possession.audio import AudioCache
    from relational_possession.data import GraphCache
    from relational_possession.experiment import ensemble,decode_scores
    from learned_possession.__main__ import write_labels
    import numpy as np
    base=FeatureCache.load(cache/'features.npz');audio=AudioCache.load(cache/'audio_cache.npz');graph=GraphCache.load(cache/'graph.npz')
    if len(base.x)!=manifest['n_frames'] or base.fps!=25 or list(map(list,base.halves))!=manifest['halves']:
        raise ValueError('Input video clock differs from the saved match model')
    print('Running the three saved graph/TCN seeds; no labels or stored predictions are read.',flush=True)
    p,q,info=ensemble(HERE/'model/graph_tcn',base,audio,graph,args.device)
    labels=decode_scores(p,q,base,info['durations'],switch_cost=8)
    if np.any(labels[base.graded_mask()]=='X') or np.any(labels[~base.graded_mask()]!='X'):
        raise ValueError('Invalid graded-frame coverage')
    write_labels(args.out,labels);actual=sha(args.out)
    report=dict(prediction_sha256=actual,expected_sha256=manifest['prediction_sha256'],matches_preserved_candidate=actual==manifest['prediction_sha256'],
        n_frames=len(labels),device=args.device,input_mode='original_assets' if args.data_dir else 'saved_observation_caches',
        target_reference_labels_used=False,stored_predictions_read=False)
    Path(str(args.out)+'.verification.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['matches_preserved_candidate']:raise RuntimeError('Inference completed but differs from the preserved CSV; see verification JSON. No predictions were corrected.')
    print('Verified byte-identical prediction CSV:',str(Path(args.out).resolve()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group();group.add_argument('--data-dir');group.add_argument('--cache-dir')
    parser.add_argument('--work-dir');parser.add_argument('--out',default='submission_generated.csv')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    run(parser.parse_args())
