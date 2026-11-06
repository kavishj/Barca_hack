import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import types
import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from visual_possession.internvideo import (FrozenVideo,vision_definitions,sdpa_attention,load)
from visual_possession.media import clip_frames,decode_clips,normalize_rgb
from visual_possession.cache import VisualCache,VisualNormalizer
from visual_possession.geometry import plan_views,crop_views
from visual_possession.model import VisionNet,VisualGraph
from relational_possession.model import RelationalNet,Config
from relational_possession.engine import tensors
from temporal_possession.network import deterministic
from test_relational_possession import graph_fixture

SOURCE=Path(__file__).resolve().parents[1]/'third_party/internvideo2_source.py'


def tiny_encoder():
    return vision_definitions(SOURCE)['PretrainInternVideo2'](img_size=28,patch_size=14,
        embed_dim=32,depth=2,num_heads=4,mlp_ratio=2,clip_embed_dim=16,
        attn_pool_num_heads=4,num_frames=4,clip_teacher_embed_dim=32,
        clip_teacher_final_dim=16,clip_return_layer=2,sep_image_video_pos_embed=True,
        use_flash_attn=False,use_fused_rmsnorm=False,use_fused_mlp=False,use_checkpoint=False)


class FrozenVisionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_standard_attention_and_pool_only_match_released_feature_path(self):
        torch.manual_seed(7)
        original=tiny_encoder().eval()
        projection=nn.Linear(16,12).eval()
        adapted=copy.deepcopy(original)
        for block in adapted.blocks:block.attn._naive_attn=types.MethodType(sdpa_attention,block.attn)
        model=FrozenVideo(adapted,copy.deepcopy(projection)).eval()
        frames=torch.randn(2,4,3,28,28)
        with torch.inference_mode():
            expected=F.normalize(projection(original(frames.permute(0,2,1,3,4))[1]).float(),dim=-1)
            actual=model(frames)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)

    def test_source_hash_prevents_silent_remote_code_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.py';source.write_text('raise RuntimeError("unexpected code")')
            with self.assertRaisesRegex(ValueError,'reviewed pinned revision'):vision_definitions(source)

    def test_selective_frozen_loader_reproduces_tensors_and_ignores_text(self):
        from safetensors.torch import save_file
        torch.manual_seed(2)
        reference=FrozenVideo(tiny_encoder(),nn.Linear(16,12))
        state=reference.state_dict()|{'text_encoder.unused':torch.randn(3)}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            save_file(state,root/'model.safetensors')
            (root/'config.json').write_text(json.dumps(dict(model=dict(
                vision_encoder=dict(clip_embed_dim=16),embed_dim=12))))
            (root/'model.safetensors.index.json').write_text(json.dumps(dict(
                weight_map={key:'model.safetensors' for key in state})))
            with patch('visual_possession.internvideo.download',return_value=root),patch(
                    'visual_possession.internvideo.make_encoder',side_effect=lambda *args:tiny_encoder()):
                actual,info=load('unused',devices=['cpu'])
            self.assertTrue(info['frozen'])
            self.assertTrue(all(not p.requires_grad for p in actual.parameters()))
            for name,value in actual.state_dict().items():
                torch.testing.assert_close(value,state[name].half(),rtol=0,atol=0)
            self.assertNotIn('text_encoder.unused',actual.state_dict())


class VideoMediaTests(unittest.TestCase):
    def test_clips_are_clamped_to_their_half_and_rgb_channels_are_preserved(self):
        np.testing.assert_array_equal(clip_frames(100,((1,100,200),)),[100,100,106,118])
        np.testing.assert_array_equal(clip_frames(200,((1,100,200),)),[182,194,200,200])
        with self.assertRaises(ValueError):clip_frames(80,((1,100,200),))
        image=np.zeros((4,20,30,3),np.uint8);image[...,0]=255
        normalized=normalize_rgb(image)
        self.assertEqual(tuple(normalized.shape),(4,3,224,224))
        self.assertAlmostEqual(float(normalized[0,0,0,0]),(1-.485)/.229,places=5)
        self.assertAlmostEqual(float(normalized[0,2,0,0]),-.406/.225,places=5)

    def test_pts_decoding_preserves_overlapping_requested_frame_indices(self):
        import av
        with tempfile.TemporaryDirectory() as folder:
            video=Path(folder)/'grid.mkv'
            with av.open(str(video),'w') as output:
                stream=output.add_stream('ffv1',rate=25);stream.width=64;stream.height=48;stream.pix_fmt='bgr0'
                for index in range(100):
                    pixels=np.full((48,64,3),index,np.uint8)
                    frame=av.VideoFrame.from_ndarray(pixels,format='rgb24')
                    frame.pts=index;frame.time_base=Fraction(1,25)
                    for packet in stream.encode(frame):output.mux(packet)
                for packet in stream.encode():output.mux(packet)
            centers=[1,20,45,70,99]
            clips=list(decode_clips(video,centers,((1,0,99),)))
            for (center,rgb),expected_center in zip(clips,centers):
                self.assertEqual(center,expected_center)
                np.testing.assert_array_equal(rgb[:,0,0,0],clip_frames(center,((1,0,99),)))


class VisualFusionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def fixture(self):
        graph=graph_fixture(500,((1,0,199),(2,300,499)))
        centers=np.r_[np.arange(0,200,10),199,np.arange(300,500,10),499]
        half_ids=np.where(centers<200,1,2).astype(np.int8)
        rng=np.random.default_rng(7)
        embedding=rng.normal(size=(len(centers),3,8)).astype(np.float32)
        available=np.ones((len(centers),3),np.float32);available[3:6,1]=0
        return graph,VisualCache(centers,half_ids,embedding,available,graph.halves,25,{})

    def test_visual_pca_fits_training_only_and_team_swap_uses_shared_components(self):
        graph,cache=self.fixture();selected=np.zeros(500,bool);selected[:150]=True
        before=VisualNormalizer.fit(cache,selected,3)
        altered=copy.deepcopy(cache);altered.embeddings[~selected[cache.centers]]+=100000
        after=VisualNormalizer.fit(altered,selected,3)
        np.testing.assert_array_equal(before.means,after.means)
        np.testing.assert_array_equal(before.components,after.components)
        global_a,pair_a=before.transform(cache,500)
        swapped=copy.deepcopy(cache)
        swapped.embeddings[:,[1,2]]=swapped.embeddings[:,[2,1]]
        swapped.availability[:,[1,2]]=swapped.availability[:,[2,1]]
        global_b,pair_b=before.transform(swapped,500)
        np.testing.assert_array_equal(global_a,global_b)
        np.testing.assert_array_equal(pair_a[:,[1,0]],pair_b)
        self.assertTrue(np.all(global_a[200:300]==0))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vision.npz';before.save(path);loaded=VisualNormalizer.load(path)
            np.testing.assert_array_equal(before.components,loaded.components)

    def test_zero_gate_matches_original_graph_weights_outputs_and_rng(self):
        config=Config(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=0)
        deterministic(4);original=RelationalNet(6,4,15,config,'graph_tcn');old_rng=torch.get_rng_state()
        deterministic(4);fusion=VisionNet(6,4,15,config,5,enabled=False)
        torch.testing.assert_close(torch.get_rng_state(),old_rng,rtol=0,atol=0)
        for name,value in original.state_dict().items():
            torch.testing.assert_close(value,fusion.state_dict()[name],rtol=0,atol=0)
        graph=graph_fixture(130,((1,0,129),));data=tensors(graph,0,130,1,'cpu')
        global_x=torch.randn(130,6);pair_x=torch.randn(130,2,4)
        original.eval();fusion.eval()
        with torch.inference_mode():
            for expected,actual in zip(original(global_x,pair_x,data),fusion(global_x,pair_x,data)):
                torch.testing.assert_close(expected,actual,rtol=0,atol=0)

    def test_fusion_preserves_team_symmetry_with_nonzero_visual_weights(self):
        deterministic(1);config=Config(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=0)
        model=VisionNet(6,4,15,config,5).eval()
        with torch.no_grad():
            model.visual_global.weight.normal_();model.visual_pair.weight.normal_()
        graph=graph_fixture(130,((1,0,129),));data=tensors(graph,0,130,1,'cpu')
        data['visual_global']=torch.randn(130,5);data['visual_global'][:,-1]=1
        data['visual_pair']=torch.randn(130,2,5);data['visual_pair'][:,:,-1]=1
        global_x=torch.randn(130,6);pair_x=torch.randn(130,2,4)
        swapped={key:value.clone() for key,value in data.items()}
        swapped['roles']=torch.where(swapped['roles']<2,1-swapped['roles'],swapped['roles'])
        swapped['visual_pair']=swapped['visual_pair'][:,[1,0]]
        with torch.inference_mode():
            a,b,_=model(global_x,pair_x,data)
            c,d,_=model(global_x,pair_x[:,[1,0]],swapped)
        torch.testing.assert_close(a[:,[1,0,2]],c,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(b,d,atol=2e-6,rtol=2e-5)

    def test_disabled_visual_training_matches_original_graph_optimization(self):
        from relational_possession.engine import train_seed as old_train
        from visual_possession.engine import train_seed as new_train
        deterministic(2);rng=np.random.default_rng(2)
        gx=rng.normal(size=(500,6)).astype(np.float32);px=rng.normal(size=(500,2,4)).astype(np.float32)
        graph=graph_fixture(500,((1,0,499),))
        joined=VisualGraph(graph,np.ones((500,5),np.float32),np.ones((500,2,5),np.float32))
        codes=np.full(500,3,np.int8);codes[100:300]=np.repeat([0,1,2,0],50);selected=codes!=3
        config=Config(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=.15,warm_seconds=1,epochs=2)
        old,_=old_train(gx,px,graph,codes,selected,config,'graph_tcn',3,'cpu')
        new,_=new_train(gx,px,joined,codes,selected,config,3,'cpu',enabled=False)
        for name,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[name],rtol=0,atol=0)

    def test_crop_planning_uses_normalized_pitch_and_rejects_stale_ball(self):
        count=120;players=np.full((count,46,2),np.nan,np.float32)
        players[:,0]=[40,30];players[:,23]=[45,30]
        ball=np.tile([42,30.],(count,1));ball[20:]=np.nan
        roles=np.tile(np.repeat([0,1],23),(count,1))
        homography=np.tile(np.diag([1000.,500.,1.]),(count,1,1))
        match=types.SimpleNamespace(players=players,ball=ball,homography=homography,pitch=(100.,50.),fps=25,
            halves=((1,0,119),),team_by_frame=lambda:roles)
        geometry=plan_views(match,1000,500,step=25,side=256,max_anchor_age=25)
        np.testing.assert_array_equal(geometry.slots[0],[0,23])
        left,top,right,bottom=geometry.boxes[0,0,0]
        self.assertEqual((left+right)//2,400)
        self.assertEqual((top+bottom)//2,268)
        self.assertFalse(geometry.valid[-1].any())
        rgb=np.full((4,500,1000,3),80,np.uint8)
        crops,availability=crop_views(rgb,geometry.boxes[0],geometry.valid[0])
        self.assertEqual(crops[0].shape,(4,256,256,3));self.assertTrue(np.all(availability==1))

    def test_complete_visual_study_reuses_control_and_reloads_without_labels(self):
        from test_temporal_audio import fixture
        from learned_possession.features import FeatureCache
        from learned_possession.__main__ import write_labels
        from temporal_possession.audio import sha
        from relational_possession.experiment import compare as control_compare
        from visual_possession.experiment import compare,predict
        base,audio=fixture(10000)
        base=FeatureCache(np.column_stack([base.x,np.ones((10000,2),np.float32)]),
            (*base.names,'home_nearest_1','away_nearest_1'),25,base.halves,{})
        graph=graph_fixture(10000,base.halves)
        centers=np.concatenate([np.unique(np.r_[np.arange(first,last+1,50),last]) for _,first,last in base.halves])
        ids=np.concatenate([np.full(len(np.unique(np.r_[np.arange(first,last+1,50),last])),half,np.int8)
                            for half,first,last in base.halves])
        rng=np.random.default_rng(3)
        vision=VisualCache(centers,ids,
            rng.normal(size=(len(centers),3,32)).astype(np.float32),np.ones((len(centers),3),np.float32),base.halves,25,{})
        train=np.full(10000,'X');validation=np.full(10000,'X')
        train[100:300]=np.repeat(['H','A','D','H'],50);train[3500:3700]=train[100:300]
        validation[8000:8150]=np.repeat(['H','A','D'],50)
        def small(**kwargs):
            settings=dict(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=0,warm_seconds=1)
            settings.update(kwargs)
            return Config(**settings)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            fp,ap,gp,vp=(root/name for name in ('features.npz','audio.npz','graph.npz','visual.npz'))
            base.save(fp);audio.save(ap);graph.save(gp);vision.save(vp)
            tr,va=root/'train.csv',root/'validation.csv';write_labels(tr,train);write_labels(va,validation)
            audit=root/'audit.json';audit.write_text(json.dumps(dict(train_labels_sha256=sha(tr),validation_labels_sha256=sha(va))))
            source=root/'control'
            with patch('relational_possession.experiment.Config',side_effect=small):
                control_compare(Namespace(features=str(fp),audio=str(ap),graph=str(gp),train_labels=str(tr),
                    validation_labels=str(va),audit=str(audit),out=str(source),epochs=1,seeds=[0],
                    device='cpu',bootstrap=0,compare=[],cross_validate=True,calibrate=False))
            study=root/'visual_study'
            args=Namespace(features=str(fp),audio=str(ap),graph=str(gp),vision=str(vp),train_labels=str(tr),
                validation_labels=str(va),source_study=str(source),out=str(study),epochs=1,seeds=[0],device='cpu',bootstrap=0)
            with patch('visual_possession.experiment.Config',side_effect=small):compare(args)
            self.assertEqual((study/'global/control.csv').read_bytes(),(source/'global/graph_tcn/predictions.csv').read_bytes())
            for name in ('audio_normalizer.npz','symmetric.npz'):
                self.assertEqual((study/'global/preprocessing'/name).read_bytes(),(source/'global/preprocessing'/name).read_bytes())
            for calibrated in (False,True):
                output=root/'regenerated.csv'
                with patch('visual_possession.experiment.read_labels',side_effect=AssertionError('Reference read')):
                    predict(Namespace(features=str(fp),audio=str(ap),graph=str(gp),vision=str(vp),
                        model=str(study/'global/graph_vision'),out=str(output),device='cpu',calibrated=calibrated))
                filename='calibrated.csv' if calibrated else 'predictions.csv'
                self.assertEqual(output.read_bytes(),(study/'global/graph_vision'/filename).read_bytes())


if __name__=='__main__':unittest.main()
