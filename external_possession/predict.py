"""Recompute the external-owner candidate from observations and saved weights only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from learned_possession.features import FeatureCache
from learned_possession.__main__ import write_labels
from temporal_possession.audio import AudioCache,sha
from temporal_possession.network import deterministic
from relational_possession.data import GraphCache
from relational_possession.model import SpatialEncoder,Config
from relational_possession.experiment import ensemble
from boosted_possession.experiment import decoded
from .direct import embed,fine_scores
from .direct_study import state


def predict(args):
    folder=Path(args.model);info=json.loads((folder/'deployment.json').read_text())
    for name in ('features','audio','graph'):
        if sha(getattr(args,name))!=info['inputs_sha256'][name]:raise ValueError('Deployment observations changed')
    for name,digest in info['model_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('Deployment weights or preprocessing changed')
    deterministic(0)
    base,audio,graph=FeatureCache.load(args.features),AudioCache.load(args.audio),GraphCache.load(args.graph)
    gp,gq,model=ensemble(folder/'graph_tcn',base,audio,graph,args.device)
    encoder=SpatialEncoder(15,Config()).to(args.device)
    encoder.load_state_dict(torch.load(folder/'encoder.pt',map_location=args.device,weights_only=True));encoder.eval()
    difference=embed(encoder,{name:getattr(graph,name) for name in ('nodes','roles','valid','positioned')},args.device)
    with np.load(folder/'probe.npz',allow_pickle=False) as data:owner=expit(difference.astype(np.float64)@data['coef'])
    owner,boundary,available=fine_scores(graph,owner,base)
    p,q=state(str(info['physical_weight']),(gp,gq),None,owner,boundary,available)
    write_labels(args.out,decoded(p,q,base,info['switch_cost'],model['durations']))
    print('DIRECT inference from weights; no reference labels:',sha(args.out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','features','audio','graph','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu');predict(parser.parse_args())
