from argparse import Namespace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from boosted_possession.features import build,swap
from boosted_possession.model import Config,fit,predict
from learned_possession.features import FeatureCache


class BoostedTests(unittest.TestCase):
    def test_automatic_tag_offsets_isolate_halves_and_rename_teams(self):
        from boosted_possession.tags import enrich
        rich=FeatureCache(np.zeros((1000,1)),('activity',),25,((1,0,399),(2,600,999)),{})
        observations=[('home','throw_in',20,90),('away','build_up',310,380),('away','throw_in',610,690)]
        a=enrich(rich,observations)
        b=enrich(rich,[('away' if club=='home' else 'home',name,first,stop) for club,name,first,stop in observations])
        np.testing.assert_array_equal(swap(a.x,a.names),b.x)
        self.assertEqual(a.x[30,a.names.index('home_auto_throw_in_active')],1)
        self.assertAlmostEqual(a.x[30,a.names.index('home_auto_throw_in_start_offset_s')],.4)
        self.assertTrue(np.isnan(a.x[600,a.names.index('home_auto_throw_in_start_offset_s')]))

    def test_equal_ensemble_preserves_distributions_and_rejects_invalid_scores(self):
        from boosted_possession.ensemble import average
        scores=dict(a=(np.tile([.8,.1,.1],(8,1)),np.zeros(8)),
                    b=(np.tile([.2,.4,.4],(8,1)),np.ones(8)))
        p,q=average(scores,('a','b'))
        np.testing.assert_allclose(p,np.tile([.5,.25,.25],(8,1)))
        np.testing.assert_array_equal(q,np.ones(8)*.5)
        with self.assertRaises(ValueError):average(scores,())
        with self.assertRaises(ValueError):average(scores,('missing',))
        with self.assertRaises(ValueError):average({'bad':(np.ones((8,3)),np.zeros(8))},('bad',))

    def test_context_does_not_cross_halves_and_team_map_is_involutive(self):
        names=('activity','home_speed_mean','away_speed_mean','contact_last_team','contact_age_s',
               'distance_margin','known_dead_from_metadata')
        x=np.tile([1,2,3,1,4,5,0],(1000,1)).astype(np.float32);x[500:,:]=100
        rich=FeatureCache(x,names,25,((1,0,399),(2,600,999)),{})
        values=build(rich,np.zeros((1000,0)),())
        self.assertNotIn('known_dead_from_metadata',values.names)
        self.assertTrue(np.isnan(values.x[398,values.names.index('activity@1')]))
        self.assertTrue(np.isnan(values.x[600,values.names.index('activity@-1')]))
        np.testing.assert_array_equal(swap(swap(values.x,values.names),values.names),values.x)
        changed=rich.x.copy();changed[400:600]=99999
        alt=build(FeatureCache(changed,names,25,rich.halves,{}),np.zeros((1000,0)),())
        np.testing.assert_array_equal(values.x[rich.graded_mask()],alt.x[rich.graded_mask()])

    def test_factorization_remains_symmetric_and_unknown_targets_do_not_fit(self):
        rng=np.random.default_rng(2);names=('home_signal','away_signal','activity','contact_last_team')
        x=rng.normal(size=(500,4)).astype(np.float32)
        c=FeatureCache(x,names,25,((1,0,499),),{})
        codes=np.tile(np.repeat([0,1,2,0],25),5).astype(np.int8);codes[400:]=3
        model=fit(c,codes,Config(max_iter=8,min_samples_leaf=2),step=1)
        p,q=predict(model,c);other=FeatureCache(swap(x,names),names,25,c.halves,{})
        a,b=predict(model,other)
        np.testing.assert_allclose(p[:,[1,0,2]],a,atol=1e-7,rtol=1e-7)
        np.testing.assert_array_equal(q,b);np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-7)
        self.assertEqual(model['counts']['dead']['frames'],400)
        changed=x.copy();changed[400:]=np.nan
        alternate=fit(FeatureCache(changed,names,25,c.halves,{}),codes,Config(max_iter=8,min_samples_leaf=2),step=1)
        for name in ('dead','owner','boundary'):
            np.testing.assert_array_equal(model[name].predict_proba(x),alternate[name].predict_proba(x))

    def test_selection_precedes_main_score_and_saved_model_regenerates_without_references(self):
        from test_temporal_audio import fixture
        from temporal_possession.network import Normalizer
        from temporal_possession.audio import sha
        from temporal_possession.experiment import metrics
        from learned_possession.__main__ import write_labels
        from learned_possession.model import near,training_mask
        from boosted_possession.experiment import compare,regenerate
        base,audio=fixture(10000)
        train=np.full(10000,'X');valid=np.full(10000,'X')
        train[100:400]=np.repeat(['H','A','D'],100);train[3500:3800]=train[100:400]
        valid[8000:8300]=train[100:400]
        codes=np.array([{'H':0,'A':1,'D':2,'X':3}[x] for x in train],np.int8)
        truth=np.array([{'H':0,'A':1,'D':2,'X':3}[x] for x in valid],np.int8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fp,ap,tp,vp=(root/n for n in ('base.npz','audio.npz','train.csv','valid.csv'))
            base.save(fp);audio.save(ap);write_labels(tp,train);write_labels(vp,valid)
            source=root/'source';source.mkdir()
            (source/'experiment.json').write_text(json.dumps(dict(train_sha256=sha(tp),validation_sha256=sha(vp),
                fingerprints=dict(feature_sha256=sha(fp),audio_sha256=sha(ap)))))
            for index in (-1,0,1):
                folder=source/('global' if index<0 else 'cv/fold_'+str(index));(folder/'preprocessing').mkdir(parents=True)
                chosen=codes!=3
                if index>=0:
                    first,stop=((100,400),(3500,3800))[index]
                    held=np.full(10000,3,np.int8);held[first:stop]=codes[first:stop]
                    remainder=codes.copy();remainder[first:stop]=3
                    chosen,_=training_mask(base,remainder,held,1500);chosen&=~near(truth!=3,1500)
                Normalizer.fit(base,audio,chosen).save(folder/'preprocessing/audio_normalizer.npz')
                model=folder/'graph_tcn';model.mkdir()
                (model/'model.json').write_text(json.dumps(dict(durations=[None]*3)))
                p=np.tile([.3,.3,.4],(10000,1)).astype(np.float32)
                np.savez_compressed(model/'scores.npz',probability=p,boundary=np.ones(10000,np.float32)*.01)
            out=root/'study';calls=[]
            def guarded(actual,prediction,bootstrap=0):
                if np.array_equal(actual,truth):
                    self.assertTrue((out/'decision.json').is_file());calls.append(True)
                return metrics(actual,prediction,bootstrap)
            def tiny(features,codes,threads):return fit(features,codes,Config(max_iter=5,min_samples_leaf=2),threads=threads)
            with patch('boosted_possession.experiment.fit',side_effect=tiny),patch('boosted_possession.experiment.metrics',side_effect=guarded):
                compare(Namespace(features=str(fp),rich_features=str(fp),audio=str(ap),train_labels=str(tp),
                    validation_labels=str(vp),source_study=str(source),out=str(out),threads=2,bootstrap=0))
            self.assertEqual(len(calls),1)
            report=json.loads((out/'report.json').read_text());self.assertEqual(len(report['decision']['oof_scores']),9)
            result=root/'regenerated.csv'
            with patch('boosted_possession.experiment.read_labels',side_effect=AssertionError('Reference read during inference')):
                regenerate(Namespace(features=str(fp),rich_features=str(fp),audio=str(ap),model=str(out/'global'),out=str(result),threads=2))
            self.assertEqual(result.read_bytes(),(out/'selected_predictions.csv').read_bytes())


if __name__=='__main__':unittest.main()
