"""Pinned frozen CLAP embeddings and predeclared prompt-relative whistle scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .audio import AudioCache,decode_audio,covered_fraction,sha
from .network import deterministic

MODEL='laion/clap-htsat-unfused'
REVISION='8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a'
PROMPTS=(
    'The sound of a referee blowing a whistle during a soccer match.',
    'The sound of a person whistling with their mouth.',
    'The sound of a crowd cheering in a soccer stadium.',
    'The sound of a commentator speaking during a soccer match.',
    'The sound of a soccer ball being kicked.',
    'The ambient sound of a soccer stadium without a whistle.')


def clip_at(wave,location,first,last,fps=25,rate=48000):
    center=round((location+.5)/fps*rate);start,stop=center-rate,center+rate
    a,b=max(start,round(first/fps*rate)),min(stop,round((last+1)/fps*rate))
    clip=np.zeros(rate*2,np.float32)
    if b>a:clip[a-start:b-start]=wave[a:b]
    return clip,a,b


def prompt_scores(audio,text,log_scale):
    if audio.ndim!=2 or text.ndim!=2 or audio.shape[1]!=text.shape[1] or len(text)!=6:
        raise ValueError('CLAP prompt feature schema differs')
    return ((audio@text.T)*log_scale.exp()).softmax(dim=1)[:,:2]


def extract(args):
    from huggingface_hub import snapshot_download
    from transformers import ClapModel,ClapProcessor
    out=Path(args.out)
    if out.exists():raise ValueError('Use a fresh CLAP audio cache')
    old=AudioCache.load(args.reference_audio);deterministic(0)
    folder=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=['config.json','preprocessor_config.json',
        'pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt']))
    processor=ClapProcessor.from_pretrained(folder,local_files_only=True)
    if processor.feature_extractor.sampling_rate!=48000:raise ValueError('Pinned CLAP sample rate changed')
    model=ClapModel.from_pretrained(folder,local_files_only=True).to(args.device).eval()
    for p in model.parameters():p.requires_grad_(False)
    with torch.inference_mode():
        text_inputs=processor(text=list(PROMPTS),return_tensors='pt',padding=True)
        text=model.get_text_features(**{k:v.to(args.device) for k,v in text_inputs.items()})
    out.parent.mkdir(parents=True,exist_ok=True);wave_path=out.parent/'clap_48000.f32'
    print('CLAP decode original audio PTS at 48 kHz',flush=True)
    wave,spans=decode_audio(args.video,wave_path,args.n_frames,old.fps,sample_rate=48000)
    clips=[];embeddings=[];scores=[];availability=[];centers=[];ids=[]
    def flush():
        if not clips:return
        features=processor.feature_extractor(clips,sampling_rate=48000,return_tensors='pt',padding='repeatpad',truncation='rand_trunc')
        with torch.inference_mode():
            e=model.get_audio_features(**{k:v.to(args.device) for k,v in features.items()})
            s=prompt_scores(e,text,model.logit_scale_a)
        embeddings.append(e.cpu().numpy());scores.append(s.cpu().numpy());clips.clear()
    for half,first,last in old.halves:
        for location in old.centers[old.half_ids==half]:
            clip,a,b=clip_at(wave,float(location),first,last,old.fps)
            clips.append(clip);availability.append(covered_fraction(a,b,spans));centers.append(location);ids.append(half)
            if len(clips)>=args.batch_size:flush()
        flush();print('CLAP encoded half',half,'centers',len(centers),flush=True)
    source_files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
    provenance=dict(encoder='CLAP',model=MODEL,revision=REVISION,checkpoint_license='Apache-2.0',source_files_sha256=source_files,
        parameters=sum(p.numel() for p in model.parameters()),sample_rate=48000,clip_seconds=2,hop_seconds=.5,prompts=PROMPTS,
        score_names=['clap_referee_relative','clap_mouth_whistling_relative'],score_semantics='Six-prompt softmax; not calibrated whistle probabilities',
        target_labels_used=False,audio_finetuned=False,prompt_tuning=False,audio_spans=spans,
        timestamp_policy='Audio PTS relative to original video PTS; AAC priming clipped; zero outside each graded half',
        source_audio_sha256=sha(args.reference_audio),video_bytes=Path(args.video).stat().st_size,torch=torch.__version__)
    cache=AudioCache(np.asarray(centers),np.asarray(ids,np.int8),np.concatenate(embeddings),np.concatenate(scores),
        np.asarray(availability,np.float32),old.halves,old.fps,provenance)
    np.testing.assert_array_equal(cache.centers,old.centers);np.testing.assert_array_equal(cache.half_ids,old.half_ids)
    cache.save(out);del wave;wave_path.unlink();print('CLAP AUDIO COMPLETE',sha(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('video','reference-audio','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--n-frames',type=int,default=199319);parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=16);extract(parser.parse_args())
