"""Entity symmetry, causal memory, censored spans and complete experiment checks."""
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from learned_possession.features import FeatureCache
from learned_possession.__main__ import write_labels
from temporal_possession.audio import sha
from temporal_possession.network import deterministic
from test_temporal_audio import fixture
from relational_possession.data import GraphCache,SymmetricNormalizer,build_graph
from relational_possession.model import ARMS,Config,RelationalNet,ObservationMemory,SpatialEncoder
from relational_possession.engine import tensors,infer_seed
from relational_possession.loss import span_targets,span_loss


def graph_fixture(n,halves):
    t=np.arange(n,dtype=np.float32)
    players=np.zeros((n,46,2),np.float32)
    players[:,:,0]=np.arange(46)[None]*2+.01*t[:,None]
    players[:,:,1]=np.arange(46)[None]%10*5
    players[20:30,4]=np.nan
    ball=np.column_stack([30+.02*t,20+np.sin(t/100)]).astype(np.float32)
    ball[40:55]=np.nan
    roles=np.tile(np.repeat([0,1],23),(n,1))
    match=SimpleNamespace(ball=ball,players=players,player_speed=np.ones((n,46),np.float32),fps=25,
                          pitch=(105.3,68.),halves=halves,n_frames=n,team_by_frame=lambda:roles)
    return build_graph(match)


class RelationalDataTests(unittest.TestCase):
    def test_graph_time_interpolation_and_missing_ball_are_explicit(self):
        graph=graph_fixture(200,((1,0,79),(2,120,199)))
        window=graph.window(42,64,1)
        self.assertTrue(np.all(window['valid'][:,-1]))
        self.assertFalse(window['positioned'][0,-1])
        self.assertEqual(window['confidence'][0],0)
        with self.assertRaises(ValueError):
            graph.window(70,130,1)
        self.assertTrue(np.all((window['weight']>=0)&(window['weight']<=1)))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'graph.npz';graph.save(path)
            reloaded=GraphCache.load(path)
            np.testing.assert_array_equal(reloaded.nodes,graph.nodes)

    def test_shared_scaling_is_equivariant_and_fits_training_only(self):
        cache,audio=fixture(600)
        rng=np.random.default_rng(0)
        x=np.column_stack([cache.x,rng.normal(size=(600,3))]).astype(np.float32)
        cache=FeatureCache(x,(*cache.names,'home_nearest_1','away_nearest_1','distance_margin'),25,cache.halves,{})
        selected=np.zeros(600,bool);selected[10:200]=True
        normalizer=SymmetricNormalizer.fit(cache,audio,selected,components=4)
        a,b=normalizer.transform(cache,audio)
        swap=copy.deepcopy(cache)
        swap.x[:,[2,3]]=swap.x[:,[3,2]]
        swap.x[:,4]*=-1
        c,d=normalizer.transform(swap,audio)
        np.testing.assert_array_equal(a,c)
        np.testing.assert_allclose(b[:,::-1],d,atol=1e-6)
        changed=copy.deepcopy(cache)
        changed.x[~selected]+=10000
        second=SymmetricNormalizer.fit(changed,audio,selected,components=4)
        for key in ('global_mean','global_scale','pair_mean','pair_scale'):
            np.testing.assert_array_equal(getattr(normalizer,key),getattr(second,key))


class RelationalArchitectureTests(unittest.TestCase):
    def test_player_permutation_and_team_swap_preserve_the_required_outputs(self):
        deterministic(2)
        config=Config(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=0)
        graph=graph_fixture(130,((1,0,129),))
        data=tensors(graph,0,130,1,'cpu')
        model=RelationalNet(6,4,15,config,'graph_memory_span').eval()
        g=torch.randn(130,6);p=torch.randn(130,2,4)
        with torch.inference_mode():
            a,b,spans=model(g,p,data)
            swapped={k:v.clone() for k,v in data.items()}
            swapped['roles']=torch.where(swapped['roles']<2,1-swapped['roles'],swapped['roles'])
            c,d,other=model(g,p[:,[1,0]],swapped)
            np.testing.assert_allclose(a[:,[1,0,2]],c,atol=2e-6)
            np.testing.assert_allclose(b,d,atol=2e-6)
            np.testing.assert_allclose(spans[:,[1,0]],other,atol=2e-6)
            permutation=torch.cat([torch.randperm(46),torch.tensor([46])])
            shuffled={k:(v[:,permutation] if k in ('nodes','roles','valid','positioned') else v) for k,v in data.items()}
            e,_,_=model(g,p,shuffled)
            np.testing.assert_allclose(a,e,atol=3e-6)

    def test_memory_matches_stepwise_recurrence_and_is_causal(self):
        deterministic(0)
        memory=ObservationMemory(5,4)
        x=torch.randn(2,91,5,requires_grad=True)
        confidence=torch.rand(91)
        actual,_=memory(x,confidence)
        candidate=torch.tanh(memory.candidate(x))
        alpha=.5+.4995*torch.sigmoid(memory.retention(x))
        beta=(1-alpha)*(.05+.95*confidence[None,:,None])
        state=torch.zeros(2,4);expected=[]
        for i in range(91):
            state=(1-beta[:,i])*state+beta[:,i]*candidate[:,i]
            expected.append(state)
        expected=torch.stack(expected,dim=1)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        changed=x.detach().clone();changed[:,50:]+=100
        other,_=memory(changed,confidence)
        torch.testing.assert_close(actual[:,:50],other[:,:50],atol=1e-6,rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA determinism check needs a GPU')
    def test_cuda_memory_backward_is_deterministic_and_finite(self):
        deterministic(0)
        model=ObservationMemory(5,8).cuda()
        x=torch.randn(2,100,5,device='cuda',requires_grad=True)
        out,_=model(x,torch.rand(100,device='cuda'))
        out.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        config=Config(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=0)
        graph=graph_fixture(130,((1,0,129),))
        network=RelationalNet(4,2,15,config,'graph_memory_span').cuda()
        logits,boundary,spans=network(torch.randn(130,4,device='cuda'),
            torch.randn(130,2,2,device='cuda'),tensors(graph,0,130,1,'cuda'))
        (logits.square().mean()+boundary.square().mean()+spans.mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None))

    def test_half_inference_does_not_carry_state_across_halftime(self):
        deterministic(1)
        config=Config(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=0)
        graph=graph_fixture(200,((1,0,79),(2,120,199)))
        model=RelationalNet(4,2,15,config,'graph_memory').eval()
        g=np.random.default_rng(0).normal(size=(200,4)).astype(np.float32)
        p=np.random.default_rng(1).normal(size=(200,2,2)).astype(np.float32)
        before,_=infer_seed(model,g,p,graph,2,'cpu')
        changed=copy.deepcopy(graph);changed.nodes[changed.half_ids==2]+=100
        g[120:]+=100;p[120:]-=100
        after,_=infer_seed(model,g,p,changed,2,'cpu')
        np.testing.assert_array_equal(before[:80],after[:80])


class RelationalSpanTests(unittest.TestCase):
    def test_unknown_edges_are_censored_and_each_spell_has_equal_mass(self):
        codes=np.array([3]*3+[0]*10+[1]*20+[2]*5+[0]*4+[3]*3,np.int8)
        target,known,weight=span_targets(codes,25)
        self.assertFalse(known[3,0]);self.assertTrue(known[3,1])
        self.assertTrue(known[13].all())
        self.assertTrue(known[38,0]);self.assertFalse(known[38,1])
        self.assertAlmostEqual(float(weight[3:13].sum()),1,places=6)
        self.assertAlmostEqual(float(weight[13:33].sum()),1,places=6)
        prediction=torch.ones((len(codes),2,2),requires_grad=True)
        target=torch.from_numpy(target);known=torch.from_numpy(known);weight=torch.from_numpy(weight)
        a,b=span_loss(prediction,torch.from_numpy(codes.astype(np.int64)),target,known,weight)
        (a+b).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        revised=target.clone();revised[3:13,0]=99999;revised[38:42,1]=99999
        c,d=span_loss(prediction,torch.from_numpy(codes.astype(np.int64)),revised,known,weight)
        torch.testing.assert_close(a,c);torch.testing.assert_close(b,d)

    def test_perfect_uncensored_intervals_have_zero_span_loss(self):
        codes=np.array([2]*4+[0]*8+[1]*5+[2]*4,np.int8)
        y,m,w=span_targets(codes,25)
        prediction=torch.from_numpy(np.repeat(y[:,None,:],2,axis=1))
        a,b=span_loss(prediction,torch.from_numpy(codes.astype(np.int64)),torch.from_numpy(y),
                      torch.from_numpy(m),torch.from_numpy(w))
        self.assertAlmostEqual(float(a),0,places=6)
        self.assertAlmostEqual(float(b),0,places=6)


class RelationalExperimentTests(unittest.TestCase):
    def test_full_four_arm_study_folds_and_reference_free_prediction(self):
        from relational_possession.experiment import compare,predict
        cache,audio=fixture(10000)
        cache=FeatureCache(np.column_stack([cache.x,np.ones((10000,2))]),
                           (*cache.names,'home_nearest_1','away_nearest_1'),25,cache.halves,{})
        graph=graph_fixture(10000,cache.halves)
        train,validation=np.full(10000,'X'),np.full(10000,'X')
        pattern=np.repeat(['H','A','D','H'],50)
        train[100:300],train[3500:3700]=pattern,pattern
        validation[8000:8050],validation[8050:8100],validation[8100:8150]='H','A','D'
        def small(**kw):
            base=dict(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=0,warm_seconds=1)
            base.update(kw)
            return Config(**base)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            fp,ap,gp=root/'features.npz',root/'audio.npz',root/'graph.npz'
            cache.save(fp);audio.save(ap);graph.save(gp)
            tr,va=root/'train.csv',root/'validation.csv'
            write_labels(tr,train);write_labels(va,validation)
            audit=root/'audit.json'
            audit.write_text(json.dumps(dict(train_labels_sha256=sha(tr),validation_labels_sha256=sha(va))))
            with patch('relational_possession.experiment.Config',side_effect=small):
                compare(Namespace(features=str(fp),audio=str(ap),graph=str(gp),train_labels=str(tr),
                    validation_labels=str(va),audit=str(audit),out=str(root/'study'),epochs=1,seeds=[0],
                    device='cpu',bootstrap=0,compare=[],cross_validate=True,calibrate=True))
            report=json.loads((root/'study/report.json').read_text())
            self.assertEqual(len(report['cv']),2)
            self.assertEqual(report['experiment']['training_frames'],400)
            for arm in ARMS:
                model=root/'study/global'/arm
                output=root/(arm+'.csv')
                with patch('relational_possession.experiment.read_labels',side_effect=AssertionError('Reference read')):
                    predict(Namespace(features=str(fp),audio=str(ap),graph=str(gp),model=str(model),
                                      out=str(output),device='cpu'))
                self.assertEqual(output.read_bytes(),(model/'predictions.csv').read_bytes())
                with patch('relational_possession.experiment.read_labels',side_effect=AssertionError('Reference read')):
                    predict(Namespace(features=str(fp),audio=str(ap),graph=str(gp),model=str(model),
                                      out=str(output),device='cpu',calibrated=True))
                self.assertEqual(output.read_bytes(),(model/'calibrated.csv').read_bytes())


if __name__=='__main__':
    unittest.main()
