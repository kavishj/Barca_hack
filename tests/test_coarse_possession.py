from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from coarse_possession.model import Config,CoarseNet
from relational_possession.model import RelationalNet
from relational_possession.engine import tensors
from test_relational_possession import graph_fixture
from temporal_possession.network import deterministic


class CoarseTests(unittest.TestCase):
    def test_same_capacity_coarse_interpolation_equivariance_and_half_isolation(self):
        config=Config(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=0)
        self.assertEqual(config.radius,10)
        deterministic(0);fine=RelationalNet(4,3,15,config,'graph_tcn').eval()
        deterministic(0);model=CoarseNet(4,3,15,config,'graph_tcn').eval()
        for name,value in fine.state_dict().items():torch.testing.assert_close(value,model.state_dict()[name],atol=0,rtol=0)
        graph=graph_fixture(500,((1,0,199),(2,300,499)))
        data=tensors(graph,0,200,1,'cpu');gx=torch.randn(200,4);px=torch.randn(200,2,3)
        flipped=dict(data,roles=torch.where(data['roles']<2,1-data['roles'],data['roles']))
        with torch.no_grad():
            a,b,_=model(gx,px,data);c,d,_=model(gx,px[:,[1,0]],flipped)
            at=data['coarse_to_fine'];identity=torch.arange(len(at))
            coarse=dict(data,left=identity,right=identity,weight=torch.zeros(len(at)),coarse_to_fine=identity)
            ca,cb,_=fine(gx[at],px[at],coarse)
        torch.testing.assert_close(a[:,[1,0,2]],c,atol=2e-6,rtol=2e-5);torch.testing.assert_close(b,d,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(a[at],ca,atol=0,rtol=0);torch.testing.assert_close(b[at],cb,atol=0,rtol=0)
        before=a.clone();graph.nodes[graph.half_ids==2]=1e4
        with torch.no_grad():after=model(gx,px,tensors(graph,0,200,1,'cpu'))[0]
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        model.train();p,q,_=model(gx,px,data);(p.square().mean()+q.square().mean()).backward()
        self.assertTrue(all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()))

    def test_study_selects_before_main_and_regenerates_without_reference_files(self):
        from test_temporal_audio import fixture
        from relational_possession.data import SymmetricNormalizer
        from learned_possession.__main__ import write_labels
        from learned_possession.model import training_mask,near
        from temporal_possession.audio import sha
        from temporal_possession.experiment import metrics
        from coarse_possession.experiment import compare,predict
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
            root=Path(tmp);fp,ap,gp,tp,vp=(root/n for n in ('base.npz','audio.npz','graph.npz','train.csv','validation.csv'))
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
            with patch('coarse_possession.experiment.Config',side_effect=small),patch('coarse_possession.experiment.metrics',side_effect=guarded):
                compare(Namespace(features=str(fp),audio=str(ap),graph=str(gp),source_study=str(source),boosted_study=str(boosted),
                    train_labels=str(tp),validation_labels=str(vp),out=str(out),epochs=1,seeds=[0],device='cpu',bootstrap=0))
                result=root/'regenerated.csv'
                with patch('coarse_possession.experiment.read_labels',side_effect=AssertionError('Reference read')):
                    predict(Namespace(features=str(fp),audio=str(ap),graph=str(gp),model=str(out/'global'),out=str(result),device='cpu'))
            self.assertEqual(len(calls),1)
            self.assertEqual(result.read_bytes(),(out/'selected_predictions.csv').read_bytes())


if __name__=='__main__':unittest.main()
