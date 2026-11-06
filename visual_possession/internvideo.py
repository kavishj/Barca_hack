"""Pinned InternVideo2 vision-only inference without optional FlashAttention.

The public source is checked byte-for-byte before compiling its reviewed
vision definitions. Text, tokenisation and distillation outputs are not needed
for the released model's projected pooled video feature.
"""
import ast
from collections import defaultdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import time
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

MODEL_ID='OpenGVLab/InternVideo2-Stage2_6B'
REVISION='fd8ef758daf132d4738325d39546dd76ce26911d'
SOURCE_SHA256='9c1ddd97062e971ebd35e1645db3e06e0cdbb66cc1590777d258c67e4be9344f'


class DropPath(nn.Module):
    """Parameter-free stochastic depth; frozen feature extraction uses eval."""
    def __init__(self,drop_prob=0.):
        super().__init__();self.drop_prob=drop_prob
    def forward(self,x):
        if not self.training or self.drop_prob==0:return x
        keep=1-self.drop_prob
        mask=x.new_empty((x.shape[0],)+(1,)*(x.ndim-1)).bernoulli_(keep)
        return x*mask/keep


def vision_definitions(source):
    source=Path(source).read_bytes()
    if hashlib.sha256(source).hexdigest()!=SOURCE_SHA256:
        raise ValueError('Public InternVideo2 source differs from the reviewed pinned revision')
    tree=ast.parse(source)
    definitions=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.lineno<1093]
    namespace=dict(np=np,torch=torch,nn=nn,F=F,math=math,partial=partial,checkpoint=checkpoint,
                   DropPath=DropPath,to_2tuple=lambda x:x if isinstance(x,tuple) else (x,x),
                   trunc_normal_=nn.init.trunc_normal_)
    # The optional flash/fused branches are disabled; their imports never run.
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'pinned_internvideo2_vision','exec'),namespace)
    return namespace


def sdpa_attention(self,x):
    b,n,c=x.shape
    q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4).unbind(0)
    if self.qk_normalization:
        q=self.q_norm(q.transpose(1,2).flatten(-2,-1)).view(b,n,self.num_heads,-1).transpose(1,2)
        k=self.k_norm(k.transpose(1,2).flatten(-2,-1)).view(b,n,self.num_heads,-1).transpose(1,2)
    result=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=self.scale)
    return self.proj_drop(self.proj(result.transpose(1,2).reshape(b,n,c)))


class FrozenVideo(nn.Module):
    def __init__(self,encoder,projection):
        super().__init__();self.vision_encoder=encoder;self.vision_proj=projection
    def forward(self,frames):
        encoder=self.vision_encoder
        first=encoder.patch_embed.proj.weight
        x=encoder.patch_embed(frames.to(device=first.device,dtype=first.dtype).permute(0,2,1,3,4))
        b,t,n,c=x.shape
        x=torch.cat([encoder.cls_token.expand(b,-1,-1),x.reshape(b,t*n,c)],dim=1)+encoder.pos_embed
        for block in encoder.blocks:
            x=block(x.to(next(block.parameters()).device))
        x=x.to(next(encoder.clip_projector.parameters()).device)
        pooled=encoder.clip_projector(x)
        return F.normalize(self.vision_proj(pooled).float(),dim=-1)


def make_encoder(source,config,optimized=True):
    if any(config['model']['vision_encoder'][name] for name in ('use_flash_attn','use_fused_mlp','use_fused_rmsnorm')):
        raise ValueError('This adapter supports standard PyTorch inference only')
    namespace=vision_definitions(source)
    encoder=namespace['pretrain_internvideo2_6b_patch14_224'](types.SimpleNamespace(
        vision_encoder=types.SimpleNamespace(**config['model']['vision_encoder'])))
    if optimized:
        for block in encoder.blocks:
            block.with_cp=False
            block.attn._naive_attn=types.MethodType(sdpa_attention,block.attn)
    return encoder


def download(cache):
    from huggingface_hub import snapshot_download
    print('Downloading pinned InternVideo2-6B weights to temporary cache',flush=True)
    return Path(snapshot_download(MODEL_ID,revision=REVISION,cache_dir=cache,
                 allow_patterns=['modeling_internvideo2.py','config.json','model.safetensors.index.json','*.safetensors'],
                 max_workers=4))


def load(cache,devices=None):
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    folder=download(cache)
    config=json.loads((folder/'config.json').read_text())
    conf=config['model']['vision_encoder']
    for flag in ('use_flash_attn','use_fused_rmsnorm','use_fused_mlp','use_checkpoint'):conf[flag]=False
    devices=devices or ['cuda:'+str(i) for i in range(min(torch.cuda.device_count(),2))]
    if not devices:raise ValueError('Frozen 6B video encoder needs a CUDA GPU')
    with init_empty_weights():
        encoder=make_encoder(folder/'modeling_internvideo2.py',config)
        # These weights contribute only to distillation/image-only outputs.
        for name in ('clip_decoder','final_clip_decoder','clip_pos_embed','clip_img_pos_embed','img_pos_embed'):
            delattr(encoder,name)
        model=FrozenVideo(encoder,nn.Linear(conf['clip_embed_dim'],config['model']['embed_dim']))
    index=json.loads((folder/'model.safetensors.index.json').read_text())['weight_map']
    expected=set(model.state_dict())
    if not expected<=set(index):raise ValueError('Checkpoint lacks required vision tensors')
    grouped=defaultdict(list)
    for key in sorted(expected):grouped[index[key]].append(key)
    loaded=[];started=time.monotonic()
    for shard,keys in sorted(grouped.items()):
        with safe_open(folder/shard,framework='pt',device='cpu') as stream:
            for key in keys:
                target=devices[0]
                if key.startswith('vision_encoder.blocks.'):
                    layer=int(key.split('.')[2]);target=devices[min(layer*len(devices)//len(encoder.blocks),len(devices)-1)]
                elif key.startswith(('vision_encoder.clip_projector.','vision_proj.')):target=devices[-1]
                tensor=stream.get_tensor(key)
                set_module_tensor_to_device(model,key,target,value=tensor,dtype=torch.float16)
                loaded.append(key)
                del tensor
        print('Loaded vision shard',shard,flush=True)
    if set(loaded)!=expected or any(p.is_meta for p in model.parameters()):
        raise ValueError('Incomplete frozen vision checkpoint loading')
    model.eval().requires_grad_(False)
    info=dict(model=MODEL_ID,revision=REVISION,source_sha256=SOURCE_SHA256,
              loaded_parameters=sum(p.numel() for p in model.parameters()),
              loaded_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
              frozen=all(not p.requires_grad for p in model.parameters()),devices=devices,
              dtype='float16 weights; float32 unit-normalized features',feature_dimensions=config['model']['embed_dim'],
              loaded_tensors=len(loaded),load_seconds=time.monotonic()-started,
              implementation='Reviewed public vision definitions; torch SDPA; omit unused text/distillation outputs')
    print('Frozen video encoder ready:',json.dumps(info),flush=True)
    return model,info


def pilot(args):
    from temporal_possession.network import deterministic
    from .media import decode_clips,normalize_rgb
    deterministic(0)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    model,info=load(args.model_cache)
    centers=[12000,70000,125000,184000]
    records=[]
    for center,rgb in decode_clips(args.video,centers,((1,11185,90651),(2,110808,190197))):
        tensor=normalize_rgb(rgb)[None]
        started=time.monotonic()
        with torch.inference_mode():
            feature=model(tensor)
            repeated=model(tensor)
        for device in info['devices']:torch.cuda.synchronize(device)
        if not torch.isfinite(feature).all() or not torch.equal(feature,repeated):
            raise ValueError('Pilot feature is non-finite or fails deterministic repetition')
        records.append(dict(frame=center,seconds_two_forwards=time.monotonic()-started,
                            norm=float(feature.norm()),repeat_equal=True))
        print('Video pilot clip',center,'passed',flush=True)
    info.update(pilot=records,torch=torch.__version__,cuda=torch.version.cuda,
                peak_gpu_bytes=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],
                reference_labels_read=False,success=True)
    (out/'pilot.json').write_text(json.dumps(info,indent=2)+'\n')
    print('VIDEO GPU PILOT COMPLETE',flush=True)
