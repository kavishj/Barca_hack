"""Pinned frozen AudioSet AST embeddings and Whistle/Whistling scores."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from .audio import AudioCache,decode_audio,covered_fraction,sha
from .clap import clip_at
from .network import deterministic

MODEL='MIT/ast-finetuned-audioset-10-10-0.4593'
REVISION='f826b80d28226b62986cc218e5cec390b1096902'


def whistle_indices(id2label):
    labels={str(value).strip().casefold():int(key) for key,value in id2label.items()}
    if 'whistle' not in labels or 'whistling' not in labels:raise ValueError('Pinned AST whistle classes missing')
    return [labels['whistle'],labels['whistling']]


def extract(args):
    from huggingface_hub import snapshot_download
    from transformers import ASTFeatureExtractor,ASTForAudioClassification
    out=Path(args.out)
    if out.exists():raise ValueError('Use a fresh AST audio cache')
    old=AudioCache.load(args.reference_audio);deterministic(0)
    folder=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=['config.json','preprocessor_config.json','model.safetensors']))
    processor=ASTFeatureExtractor.from_pretrained(folder,local_files_only=True)
    if processor.sampling_rate!=16000 or processor.max_length!=1024:raise ValueError('Pinned AST preprocessing changed')
    model=ASTForAudioClassification.from_pretrained(folder,local_files_only=True).to(args.device).eval()
    for p in model.parameters():p.requires_grad_(False)
    indices=whistle_indices(model.config.id2label);out.parent.mkdir(parents=True,exist_ok=True)
    wave_path=out.parent/'ast_16000.f32';print('AST decode original audio PTS at 16 kHz',flush=True)
    wave,spans=decode_audio(args.video,wave_path,args.n_frames,old.fps,sample_rate=16000)
    clips=[];centers=[];ids=[];embeddings=[];scores=[];availability=[]
    def flush():
        if not clips:return
        inputs=processor(clips,sampling_rate=16000,return_tensors='pt')
        amp=torch.autocast('cuda',dtype=torch.float16) if str(args.device).startswith('cuda') else nullcontext()
        with torch.inference_mode(),amp:
            output=model.audio_spectrogram_transformer(input_values=inputs['input_values'].to(args.device))
            pooled=output.pooler_output;logits=model.classifier(pooled)
        embeddings.append(pooled.float().cpu().numpy());scores.append(logits.float().sigmoid()[:,indices].cpu().numpy());clips.clear()
    for half,first,last in old.halves:
        for location in old.centers[old.half_ids==half]:
            clip,a,b=clip_at(wave,float(location),first,last,old.fps,rate=16000)
            clips.append(clip);centers.append(location);ids.append(half);availability.append(covered_fraction(a,b,spans))
            if len(clips)>=args.batch_size:
                flush()
                if len(centers)%2048==0:print('AST encoded centers',len(centers),flush=True)
        flush();print('AST encoded half',half,'centers',len(centers),flush=True)
    provenance=dict(encoder='AST',model=MODEL,revision=REVISION,checkpoint_license='BSD-3-Clause',
        source_files_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()},parameters=sum(p.numel() for p in model.parameters()),
        sample_rate=16000,clip_seconds=2,hop_seconds=.5,native_fbank_frames=1024,score_names=['ast_whistle','ast_whistling'],
        score_semantics='Frozen sigmoid AudioSet Whistle/Whistling scores; not target acoustic labels',whistle_class_indices=indices,
        target_labels_used=False,audio_finetuned=False,prompt_tuning=False,audio_spans=spans,
        inference_precision='cuda_float16_autocast' if str(args.device).startswith('cuda') else 'float32',
        timestamp_policy='Audio PTS relative to original video PTS; priming clipped; zero outside graded halves',
        source_audio_sha256=sha(args.reference_audio),video_bytes=Path(args.video).stat().st_size,torch=torch.__version__)
    cache=AudioCache(np.asarray(centers),np.asarray(ids,np.int8),np.concatenate(embeddings),np.concatenate(scores),
        np.asarray(availability,np.float32),old.halves,old.fps,provenance)
    if cache.embeddings.shape[1]!=768:raise ValueError('Pinned AST pooled width changed')
    np.testing.assert_array_equal(cache.centers,old.centers);np.testing.assert_array_equal(cache.half_ids,old.half_ids)
    cache.save(out);del wave;wave_path.unlink();print('AST AUDIO COMPLETE',sha(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('video','reference-audio','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--n-frames',type=int,default=199319);parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=16);extract(parser.parse_args())
