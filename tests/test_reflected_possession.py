from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from relational_possession.model import Config
from test_relational_possession import graph_fixture
from reflected_possession.augmentation import Reflections

class ReflectionTests(unittest.TestCase):
    def test_reflections_preserve_geometry_missingness_and_inputs(self):
        norm=SimpleNamespace(audio_normalizer=SimpleNamespace(names=('ball_y','ball_vy','activity')),
            global_indices=np.arange(3),global_mean=np.array([2.,-3.,0.]),global_scale=np.array([2.,1.5,1.]))
        transform=Reflections(norm,0)
        gx=torch.tensor([[.5,1.,0.,0.,0.,0.],[0.,0.,0.,1.,1.,0.]])
        px=torch.randn(2,2,3);nodes=torch.randn(2,47,15);original=nodes.clone()
        graph={'nodes':nodes}
        a,b,g=transform.transform(gx,px,graph,True,True)
        torch.testing.assert_close(a[0,:2],torch.tensor([-2.5,3.]))
        torch.testing.assert_close(a[1],gx[1],atol=0,rtol=0)
        torch.testing.assert_close(nodes,original,atol=0,rtol=0)
        torch.testing.assert_close(torch.cdist(nodes[:,:,:2],nodes[:,:,:2]),torch.cdist(g['nodes'][:,:,:2],g['nodes'][:,:,:2]),atol=0,rtol=0)
        c,d,other=transform.transform(a,b,g,True,True)
        torch.testing.assert_close(c,gx,atol=0,rtol=0);torch.testing.assert_close(other['nodes'],nodes,atol=0,rtol=0)
        x1=Reflections(norm,7);x2=Reflections(norm,7)
        for _ in range(4):
            p=x1(gx,px,graph);q=x2(gx,px,graph)
            torch.testing.assert_close(p[0],q[0],atol=0,rtol=0);torch.testing.assert_close(p[2]['nodes'],q[2]['nodes'],atol=0,rtol=0)

    def test_study_selects_before_main_and_regenerates_without_reference_files(self):
        from test_temporal_audio import fixture
        from relational_possession.data import SymmetricNormalizer
        from learned_possession.__main__ import write_labels
        from learned_possession.model import training_mask,near
        from temporal_possession.audio import sha
        from temporal_possession.experiment import metrics
        from reflected_possession.experiment import compare,predict
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
            with patch('reflected_possession.experiment.Config',side_effect=small),patch('reflected_possession.experiment.metrics',side_effect=guarded):
                compare(Namespace(features=str(fp),audio=str(ap),graph=str(gp),source_study=str(source),boosted_study=str(boosted),
                    train_labels=str(tp),validation_labels=str(vp),out=str(out),epochs=1,seeds=[0],device='cpu',bootstrap=0))
                result=root/'regenerated.csv'
                with patch('reflected_possession.experiment.read_labels',side_effect=AssertionError('Reference read')):
                    predict(Namespace(features=str(fp),audio=str(ap),graph=str(gp),model=str(out/'global'),out=str(result),device='cpu'))
            self.assertEqual(len(calls),1)
            self.assertEqual(result.read_bytes(),(out/'selected_predictions.csv').read_bytes())


if __name__=='__main__':unittest.main()
