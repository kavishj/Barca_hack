import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from learned_possession.features import FeatureCache
from temporal_possession.audio import AudioCache,decode_audio
from temporal_possession.network import Normalizer
from temporal_possession.ast_audio import whistle_indices


class AstAudioTests(unittest.TestCase):
    def fixture(self,width):
        rng=np.random.default_rng(3);halves=((1,0,99),(2,120,219))
        centers=np.r_[np.arange(0,99,5),99,np.arange(120,219,5),219];ids=np.where(centers<100,1,2)
        base=FeatureCache(rng.normal(size=(220,2)).astype(np.float32),('motion','whistle'),25,halves,{})
        audio=AudioCache(centers,ids,rng.normal(size=(len(centers),width)).astype(np.float32),rng.uniform(size=(len(centers),2)).astype(np.float32),
            np.ones(len(centers)),halves,25,dict(score_names=['ast_whistle','ast_whistling']))
        return base,audio

    def test_768_cache_roundtrip_and_512_legacy_schema_are_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            for width,schema in ((512,1),(768,2)):
                _,audio=self.fixture(width);path=Path(d)/f'{width}.npz';audio.save(path)
                with np.load(path) as data:meta=json.loads(str(data['metadata']))
                self.assertEqual(meta['schema'],schema)
                if schema==2:self.assertEqual(meta['embedding_dim'],768)
                else:self.assertNotIn('embedding_dim',meta)
                loaded=AudioCache.load(path);np.testing.assert_array_equal(loaded.embeddings,audio.embeddings)

    def test_ast_pca_and_scalers_exclude_held_out_audio(self):
        base,audio=self.fixture(768);selected=np.arange(220)<90
        first=Normalizer.fit(base,audio,selected,components=4)
        audio.embeddings[audio.half_ids==2]+=9999;audio.scores[audio.half_ids==2]=1
        second=Normalizer.fit(base,audio,selected,components=4)
        for k in ('mean','scale','pca_mean','pca_components'):np.testing.assert_array_equal(getattr(first,k),getattr(second,k))
        self.assertEqual(first.pca_mean.shape,(768,));self.assertIn('ast_whistle',first.names)

    def test_classes_are_named_not_assumed_to_have_panns_indices(self):
        self.assertEqual(whistle_indices({'12':'Whistling',42:'Whistle',1:'Speech'}),[42,12])
        with self.assertRaisesRegex(ValueError,'missing'):whistle_indices({0:'Speech'})

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required for native-rate alignment')
    def test_native_16khz_decode_preserves_video_clock_and_aac_delay(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'delayed.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=32x32:r=25:d=2',
                '-f','lavfi','-i','sine=frequency=3800:sample_rate=48000:duration=1','-filter_complex','[1:a]asetpts=PTS+0.2/TB[a]',
                '-map','0:v','-map','[a]','-c:v','libx264','-c:a','aac','-y',str(path)],check=True)
            wave,spans=decode_audio(path,Path(d)/'wave.f32',50,25,sample_rate=16000)
            self.assertEqual(len(wave),32000)
            rms=np.sqrt(np.mean(np.array(wave).reshape(-1,160)**2,axis=1))
            onset=float(np.flatnonzero(rms>.01)[0])*.01
            self.assertTrue(.15<=onset<=.23);self.assertLess(float(rms[:12].max()),.001)
            self.assertGreater(spans[0][0],0);del wave


if __name__=='__main__':unittest.main()
