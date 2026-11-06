from argparse import Namespace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from external_possession.data import pass_labels
from external_possession.pretrain import augment,Owner,run as pretrain
from external_possession.model import FrozenNet
from relational_possession.model import Config,SpatialEncoder
from test_relational_possession import graph_fixture
from relational_possession.engine import tensors
from temporal_possession.network import deterministic

class ExternalTests(unittest.TestCase):
    def test_pass_labels_exclude_conflicts_offsides_and_cross_half_intervals(self):
        periods=np.r_[0,np.ones(19),np.full(20,2)]
        rows=[['Home','PASS','',1,1,8],['Away','PASS','',1,5,10],['Away','PASS','OFFSIDE',1,12,15],
              ['Home','PASS','',1,18,22],['Home','SHOT','',2,25,30]]
        data=pd.DataFrame(rows,columns=['Team','Type','Subtype','Period','Start Frame','End Frame'])
        labels=pass_labels(data,periods)
        np.testing.assert_array_equal(labels[1:5],np.zeros(4))
        self.assertTrue(np.all(labels[5:8]==3) and np.all(labels[8:10]==1))
        self.assertTrue(np.all(labels[10:]==3))

    def test_ball_dropout_removes_ball_relative_information_without_mutating_inputs(self):
        graph=graph_fixture(100,((1,0,99),));nodes=torch.from_numpy(graph.nodes[:4]);roles=torch.from_numpy(graph.roles[:4])
        valid=torch.from_numpy(graph.valid[:4]);positioned=torch.from_numpy(graph.positioned[:4]);original=nodes.clone()
        changed=nodes.clone();changed[:,-1,:10]+=3;changed[:,:,5:10]+=7
        deterministic(8);a=augment(nodes,roles,valid,positioned,player_drop=0,ball_drop=1)
        deterministic(8);b=augment(changed,roles,valid,positioned,player_drop=0,ball_drop=1)
        for left,right in zip(a,b):torch.testing.assert_close(left,right,atol=0,rtol=0)
        torch.testing.assert_close(nodes,original,atol=0,rtol=0)
        self.assertTrue(a[2][:,-1].all() and not a[3][:,-1].any())

    def test_external_pretraining_and_target_step_keep_encoder_frozen(self):
        graph=graph_fixture(100,((1,0,99),))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);samples=root/'samples.npz'
            np.savez_compressed(samples,nodes=graph.nodes,roles=graph.roles,valid=graph.valid,positioned=graph.positioned,
                labels=np.arange(len(graph.frames))%2,metadata=np.array(json.dumps(dict(target_match_labels_used=False))))
            pretrain(Namespace(samples=str(samples),out=str(root/'pretrained'),device='cpu'))
            checkpoint=root/'pretrained/spatial.pt';config=Config(dilations=(1,))
            model=FrozenNet(4,3,15,config,'graph_tcn',pretrained=checkpoint)
            model.train();self.assertFalse(model.spatial.training)
            before={n:v.clone() for n,v in model.spatial.state_dict().items()}
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            gx,px=torch.randn(100,4),torch.randn(100,2,3);data=tensors(graph,0,100,1,'cpu')
            p,q,_=model(gx,px,data);loss=torch.nn.functional.cross_entropy(p,torch.arange(100)%3)+q.square().mean()
            loss.backward();optimizer.step()
            self.assertTrue(all(p.grad is None for p in model.spatial.parameters()))
            self.assertTrue(any(p.grad is not None for p in model.blocks.parameters()))
            for n,v in before.items():torch.testing.assert_close(v,model.spatial.state_dict()[n],atol=0,rtol=0)

    def test_study_selects_before_main_and_regenerates_without_reference_files(self):
        from test_temporal_audio import fixture
        from relational_possession.data import SymmetricNormalizer
        from learned_possession.__main__ import write_labels
        from learned_possession.model import training_mask,near
        from temporal_possession.audio import sha
        from temporal_possession.experiment import metrics
        from external_possession.experiment import compare,predict
        base,audio=fixture(10000);graph=graph_fixture(10000,base.halves)
        train=np.full(10000,'X');validation=np.full(10000,'X')
        train[100:400]=np.repeat(['H','A','D'],100);train[3500:3800]=train[100:400];validation[8000:8300]=train[100:400]
        codes=np.array([{'H':0,'A':1,'D':2,'X':3}[s] for s in train],np.int8)
        truth=np.array([{'H':0,'A':1,'D':2,'X':3}[s] for s in validation],np.int8)
        def small(**kwargs):
            values=dict(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=0,warm_seconds=1)
            values.update(kwargs)
            return Config(**values)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pretrained=root/'spatial.pt'
            torch.save(SpatialEncoder(15,small(epochs=1)).state_dict(),pretrained)
            fp,ap,gp,tp,vp=(root/n for n in ('base.npz','audio.npz','graph.npz','train.csv','validation.csv'))
            base.save(fp);audio.save(ap);graph.save(gp);write_labels(tp,train);write_labels(vp,validation)
            source=root/'source';source.mkdir();boosted=root/'boosted';boosted.mkdir()
            original=dict(train_sha256=sha(tp),validation_sha256=sha(vp),fingerprints=dict(feature_sha256=sha(fp),audio_sha256=sha(ap),graph_sha256=sha(gp)))
            (source/'experiment.json').write_text(json.dumps(original))
            (boosted/'experiment.json').write_text(json.dumps(dict(fingerprints=dict(features=sha(fp),audio=sha(ap),train_labels=sha(tp),validation_labels=sha(vp)))))
            for index in (-1,0,1):
                rel=Path('global' if index<0 else 'cv/fold_'+str(index));folder=source/rel
                (folder/'preprocessing').mkdir(parents=True);chosen=codes!=3
                if index>=0:
                    first,stop=((100,400),(3500,3800))[index];held=np.full(10000,3,np.int8);held[first:stop]=codes[first:stop]
                    remainder=codes.copy();remainder[first:stop]=3
                    chosen,_=training_mask(base,remainder,held,1500);chosen&=~near(truth!=3,1500)
                SymmetricNormalizer.fit(base,audio,chosen).save(folder/'preprocessing')
                (folder/'graph_tcn').mkdir();(folder/'graph_tcn/model.json').write_text(json.dumps(dict(durations=[None]*3)))
                p=np.tile([.3,.3,.4],(10000,1)).astype(np.float32);q=np.full(10000,.01,np.float32)
                np.savez_compressed(folder/'graph_tcn/scores.npz',probability=p,boundary=q)
                (boosted/rel).mkdir(parents=True);np.savez_compressed(boosted/rel/'trees_scores.npz',probability=p,boundary=q)
            out=root/'study';calls=[]
            def guarded(actual,prediction,bootstrap=0):
                if np.array_equal(actual,truth):self.assertTrue((out/'decision.json').exists());calls.append(1)
                return metrics(actual,prediction,bootstrap)
            with patch('external_possession.experiment.Config',side_effect=small),patch('external_possession.experiment.metrics',side_effect=guarded):
                compare(Namespace(features=str(fp),audio=str(ap),graph=str(gp),source_study=str(source),boosted_study=str(boosted),pretrained=str(pretrained),
                    train_labels=str(tp),validation_labels=str(vp),out=str(out),epochs=1,seeds=[0],device='cpu',bootstrap=0))
                result=root/'regenerated.csv'
                with patch('external_possession.experiment.read_labels',side_effect=AssertionError('Reference read')):
                    predict(Namespace(features=str(fp),audio=str(ap),graph=str(gp),model=str(out/'global'),out=str(result),device='cpu'))
            self.assertEqual(len(calls),1)
            self.assertEqual(result.read_bytes(),(out/'selected_predictions.csv').read_bytes())


if __name__=='__main__':unittest.main()
