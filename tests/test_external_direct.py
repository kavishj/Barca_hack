import unittest
from argparse import Namespace
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import numpy as np
from scipy.special import expit
from external_possession.direct import probe_fit,fine_scores
from learned_possession.features import FeatureCache
from test_relational_possession import graph_fixture
from physical_possession.model import mix


class DirectExternalTests(unittest.TestCase):
    def test_matched_study_uses_decision_without_main_score_and_selects_before_validation(self):
        from external_possession.direct_study import run
        from learned_possession.__main__ import write_labels
        from temporal_possession.audio import sha
        from temporal_possession.experiment import metrics
        from boosted_possession.experiment import decoded
        from test_temporal_audio import fixture
        base,_=fixture(10000);train=np.full(10000,'X');validation=train.copy()
        train[100:400]=np.repeat(['H','A','D'],100);train[3500:3800]=train[100:400];validation[8000:8300]=train[100:400]
        truth=np.array([{'H':0,'A':1,'D':2,'X':3}[v] for v in validation],np.int8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fp,tp,vp=(root/n for n in ('features.npz','train.csv','validation.csv'))
            base.save(fp);write_labels(tp,train);write_labels(vp,validation)
            original,matched,trees=(root/n for n in ('original','matched','trees'))
            for p in (original,matched,trees):p.mkdir()
            fingerprints=dict(train_labels=sha(tp),validation_labels=sha(vp))
            (original/'experiment.json').write_text(json.dumps(dict(train_sha256=sha(tp),validation_sha256=sha(vp),fingerprints=dict(graph_sha256='synthetic'))))
            (matched/'experiment.json').write_text(json.dumps(dict(seeds=[0,1,2],fingerprints=fingerprints)))
            (matched/'decision.json').write_text(json.dumps(dict(selected='graph__8',validation_used=False)))
            p=np.tile([.45,.45,.1],(10000,1)).astype(np.float32);q=np.ones(10000,np.float32)*.01
            for relative in ('global','cv/fold_0','cv/fold_1'):
                (original/relative/'graph_tcn').mkdir(parents=True);(trees/relative).mkdir(parents=True)
                np.savez_compressed(original/relative/'graph_tcn/scores.npz',probability=p,boundary=q)
                (original/relative/'graph_tcn/model.json').write_text(json.dumps(dict(durations=[None]*3)))
                np.savez_compressed(trees/relative/'trees_scores.npz',probability=p,boundary=q)
                if relative!='global':
                    (matched/relative/'graph_tcn').mkdir(parents=True)
                    np.savez_compressed(matched/relative/'graph_tcn/scores.npz',probability=p,boundary=q)
                    write_labels(matched/relative/'graph__8.csv',decoded(p,q,base,8,[None]*3,1))
            owner=root/'owner.npz';metadata=dict(target_match_labels_used=False,feature_sha256=sha(fp),graph_sha256='synthetic')
            np.savez_compressed(owner,home=np.full(10000,.5),boundary=np.zeros(10000),available=np.ones(10000,bool),metadata=np.array(json.dumps(metadata)))
            out=root/'study';calls=[]
            def guarded(actual,prediction,bootstrap=0):
                if np.array_equal(actual,truth):self.assertTrue((out/'decision.json').exists());calls.append(1)
                return metrics(actual,prediction,bootstrap)
            with patch('external_possession.direct_study.metrics',side_effect=guarded):
                run(Namespace(features=str(fp),train_labels=str(tp),validation_labels=str(vp),owner_scores=str(owner),source_study=str(original),
                    matched_study=str(matched),boosted_study=str(trees),out=str(out),bootstrap=0))
            self.assertEqual(len(calls),1)
            report=json.loads((out/'report.json').read_text());self.assertEqual(len(report['decision']['oof_scores']),13)
            with np.load(out/'selected_scores.npz') as d:labels=decoded(d['probability'],d['boundary'],base,float(report['selected'].split('__')[1]),report['durations'])
            write_labels(out/'regenerated.csv',labels)
            self.assertEqual((out/'regenerated.csv').read_bytes(),(out/'selected_predictions.csv').read_bytes())

    def test_probe_is_antisymmetric_and_rejects_unknown_supervision(self):
        x=np.random.default_rng(5).normal(size=(100,32));y=np.where(x[:,0]>0,0,1)
        coef,iterations=probe_fit(x,y);self.assertLess(iterations,2000)
        np.testing.assert_allclose(expit(-x@coef),1-expit(x@coef),atol=1e-15)
        with self.assertRaises(ValueError):probe_fit(x,np.full(100,3))

    def test_owner_interpolation_and_boundary_variation_are_half_local_and_team_symmetric(self):
        graph=graph_fixture(500,((1,0,199),(2,300,499)))
        cache=FeatureCache(np.ones((500,1)),('ball_visible',),25,graph.halves,{})
        p=np.where(graph.frames<100,.99,.01);owner,q,seen=fine_scores(graph,p,cache)
        alternative,b,other=fine_scores(graph,1-p,cache)
        np.testing.assert_allclose(alternative,1-owner,atol=1e-7);np.testing.assert_allclose(q,b,atol=1e-7)
        changed=p.copy();changed[graph.half_ids==2]=.9
        a,c,_=fine_scores(graph,changed,cache);np.testing.assert_array_equal(a[:200],owner[:200]);np.testing.assert_array_equal(c[:200],q[:200])
        self.assertGreater(float(q[97]),.9)
        cache.x[97,0]=0
        _,masked,available=fine_scores(graph,p,cache);self.assertEqual(masked[97],0);self.assertFalse(available[97])
        graph.positioned[:,-1]=False
        self.assertTrue(np.all(fine_scores(graph,p,cache)[1]==0))

    def test_expert_preserves_dead_and_missing_ball_frames(self):
        from external_possession.direct_study import state,variants
        graph=np.tile(np.array([.6,.3,.1],np.float32),(10,1));owner=np.linspace(.01,.99,10)
        observed=np.arange(10)%2==0
        result,q=state('0.75',(graph,np.zeros(10)),(graph,np.zeros(10)),owner,np.ones(10)*.2,observed)
        np.testing.assert_array_equal(result[~observed],graph[~observed]);np.testing.assert_array_equal(result[:,2],graph[:,2])
        np.testing.assert_array_equal(q[~observed],np.zeros(5))
        self.assertEqual(len(variants('graph__8')),13);self.assertEqual(len(variants('blend__2')),14)
        with self.assertRaises(ValueError):variants('unknown__1')


if __name__=='__main__':unittest.main()
