import tempfile
from pathlib import Path
import shutil
import subprocess
import unittest
import numpy as np
import torch
from temporal_possession.clap import clip_at,prompt_scores
from temporal_possession.audio import AudioCache,decode_audio
from temporal_possession.network import Normalizer
from learned_possession.features import FeatureCache


class ClapAudioTests(unittest.TestCase):
    def test_two_second_clips_use_video_frame_centers_and_half_padding(self):
        wave=np.arange(100,dtype=np.float32)+1
        clip,a,b=clip_at(wave,10,10,19,fps=10,rate=10)
        self.assertEqual((a,b),(10,20));np.testing.assert_array_equal(clip[:10],np.zeros(10))
        np.testing.assert_array_equal(clip[10:],wave[10:20])
        other=wave.copy();other[:10]=-999;other[20:]=999
        np.testing.assert_array_equal(clip_at(other,10,10,19,fps=10,rate=10)[0],clip)

    def test_relative_scores_match_fixed_six_prompt_softmax(self):
        audio=torch.tensor([[1.,0.],[0.,1.]])
        text=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[.5,.5],[-.5,-.5]])
        actual=prompt_scores(audio,text,torch.tensor(np.log(2.),dtype=torch.float32)).numpy()
        logits=2*(audio.numpy()@text.numpy().T);expected=np.exp(logits-logits.max(1,keepdims=True));expected/=expected.sum(1,keepdims=True)
        np.testing.assert_allclose(actual,expected[:,:2],atol=1e-7)
        self.assertTrue(np.all(actual.sum(1)<1))

    def test_new_score_names_preserve_old_panns_preprocessing(self):
        rng=np.random.default_rng(1);halves=((1,0,199),(2,250,449));x=rng.normal(size=(450,2)).astype(np.float32)
        base=FeatureCache(x,('motion','whistle'),25,halves,{})
        centers=np.r_[np.arange(0,199,5),199,np.arange(250,449,5),449];ids=np.where(centers<200,1,2)
        audio=AudioCache(centers,ids,rng.normal(size=(len(centers),512)).astype(np.float32),
            rng.uniform(size=(len(centers),2)).astype(np.float32),np.ones(len(centers)),halves,25,{})
        selected=np.arange(450)<190;old=Normalizer.fit(base,audio,selected,components=4)
        audio.provenance['score_names']=['clap_referee_relative','clap_mouth_whistling_relative']
        new=Normalizer.fit(base,audio,selected,components=4)
        self.assertIn('panns_whistle',old.names);self.assertIn('clap_referee_relative',new.names)
        for k in ('mean','scale','pca_mean','pca_components'):np.testing.assert_array_equal(getattr(old,k),getattr(new,k))
        np.testing.assert_array_equal(old.transform(base,audio,'learned_audio'),new.transform(base,audio,'learned_audio'))

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg needed for timestamped synthetic audio')
    def test_48khz_decode_preserves_delayed_audio_and_output_length(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'delayed.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=32x32:r=25:d=2',
                '-f','lavfi','-i','sine=frequency=3800:sample_rate=48000:duration=1','-filter_complex',
                '[1:a]asetpts=PTS+0.2/TB[a]','-map','0:v','-map','[a]','-c:v','libx264','-c:a','aac','-y',str(path)],check=True)
            wave,spans=decode_audio(path,Path(d)/'wave.f32',50,25,sample_rate=48000)
            self.assertEqual(len(wave),96000)
            energy=np.sqrt(np.mean(np.array(wave).reshape(-1,480)**2,axis=1))
            onset=float(np.flatnonzero(energy>.01)[0])*.01
            self.assertTrue(.15<=onset<=.23);self.assertLess(float(energy[:12].max()),.001)
            self.assertGreater(spans[0][0],0);del wave


if __name__=='__main__':unittest.main()
