from argparse import Namespace
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from learned_possession.features import FeatureCache
from physical_possession.model import features,fit,probability,mix
from boosted_possession.features import swap


def fixture(n=1000):
    names=['d_home','d_away','ball_speed','tag_phase','tag_restart']
    names += [club+'_'+suffix for club in ('home','away') for suffix in ('within_2m','within_5m','speed_mean',
        'moving_fraction','spread_x','spread_y','nearest_relative_speed','nearest_radial_speed')]
    names += ['distance_margin_mean_'+str(s)+'s' for s in (1,3,5)]
    names += [kind+'_'+part for kind in ('contact','reception') for part in ('last_team','age_s','team_now')]
    x=np.abs(np.random.default_rng(3).normal(size=(n,len(names)))).astype(np.float32)
    for kind in ('contact','reception'):
        x[:,names.index(kind+'_last_team')]=1
        x[:,names.index(kind+'_team_now')]=0;x[30,names.index(kind+'_team_now')]=1
    return FeatureCache(x,tuple(names),25,((1,0,n//2-11),(2,n//2+10,n-1)),{})


class PhysicalTests(unittest.TestCase):
    def test_observations_are_antisymmetric_and_future_events_stay_in_half(self):
        cache=fixture();x,names=features(cache)
        flipped=FeatureCache(swap(cache.x,cache.names),cache.names,25,cache.halves,{})
        alternate,_=features(flipped);np.testing.assert_array_equal(alternate,-x)
        changed=copy.deepcopy(cache);changed.x[510:]*=2
        other,_=features(changed);np.testing.assert_array_equal(other[:490],x[:490])
        self.assertAlmostEqual(float(x[20,names.index('contact_future_2')]),float(np.exp(-.4/2)),places=6)
        self.assertTrue(np.all(x[~cache.graded_mask()]==0))

    def test_fitting_ignores_dead_and_unknown_targets_and_mixing_preserves_dead(self):
        cache=fixture();x,_=features(cache);codes=np.full(1000,3,np.int8)
        codes[:400]=np.tile(np.repeat([0,1,2,0],25),4)
        model=fit(x,codes,cache.halves);p=probability(model,x);flipped=probability(model,-x)
        np.testing.assert_allclose(flipped,1-p,atol=1e-14)
        changed=x.copy();changed[codes>=2]=1e6
        alt=fit(changed,codes,cache.halves)
        np.testing.assert_array_equal(model['coef'],alt['coef']);np.testing.assert_array_equal(model['scale'],alt['scale'])
        graph=np.tile(np.array([.6,.3,.1],np.float32),(1000,1))
        result=mix(graph,p,.75);np.testing.assert_array_equal(result[:,2],graph[:,2])
        np.testing.assert_allclose(result.sum(axis=1),1,atol=1e-7)
        with self.assertRaises(ValueError):mix(graph,p,2)

    def test_study_selects_before_validation_and_saved_scores_regenerate(self):
        from physical_possession.experiment import run
        from learned_possession.__main__ import write_labels
        from temporal_possession.audio import sha
        from temporal_possession.experiment import metrics
        from boosted_possession.experiment import decoded
        base=fixture(10000);train=np.full(10000,'X');validation=train.copy()
        train[100:400]=np.repeat(['H','A','D'],100);train[3500:3800]=train[100:400];validation[8000:8300]=train[100:400]
        truth=np.array([{'H':0,'A':1,'D':2,'X':3}[s] for s in validation],np.int8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fp,tp,vp=(root/n for n in ('features.npz','train.csv','validation.csv'))
            base.save(fp);write_labels(tp,train);write_labels(vp,validation)
            graph=root/'graph';trees=root/'trees';graph.mkdir();trees.mkdir()
            (graph/'experiment.json').write_text(json.dumps(dict(train_sha256=sha(tp),validation_sha256=sha(vp),fingerprints=dict(feature_sha256=sha(fp)))))
            for relative in ('global','cv/fold_0','cv/fold_1'):
                (graph/relative/'graph_tcn').mkdir(parents=True);(trees/relative).mkdir(parents=True)
                p=np.tile([.45,.45,.1],(10000,1)).astype(np.float32);q=np.ones(10000,np.float32)*.01
                np.savez_compressed(graph/relative/'graph_tcn/scores.npz',probability=p,boundary=q)
                np.savez_compressed(trees/relative/'trees_scores.npz',probability=p,boundary=q)
                (graph/relative/'graph_tcn/model.json').write_text(json.dumps(dict(durations=[None]*3)))
            out=root/'study';calls=[]
            def guarded(actual,prediction,bootstrap=0):
                if np.array_equal(actual,truth):self.assertTrue((out/'decision.json').exists());calls.append(1)
                return metrics(actual,prediction,bootstrap)
            with patch('physical_possession.experiment.metrics',side_effect=guarded):
                run(Namespace(features=str(fp),rich_features=str(fp),train_labels=str(tp),validation_labels=str(vp),
                    graph_study=str(graph),boosted_study=str(trees),out=str(out),threads=2,bootstrap=0))
            self.assertEqual(len(calls),1)
            report=json.loads((out/'report.json').read_text());self.assertEqual(len(report['decision']['oof_scores']),14)
            with np.load(out/'selected_scores.npz') as data:labels=decoded(data['probability'],data['boundary'],base,float(report['selected'].split('__')[1]),report['durations'])
            write_labels(out/'regenerated.csv',labels)
            self.assertEqual((out/'regenerated.csv').read_bytes(),(out/'selected_predictions.csv').read_bytes())


if __name__=='__main__':unittest.main()
