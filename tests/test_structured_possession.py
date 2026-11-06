"""Typed-state mathematics, exact decoder paths, timing and label-free reload."""
from argparse import Namespace
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from learned_possession.features import FeatureCache
from learned_possession.__main__ import write_labels
from temporal_possession.network import NetworkConfig,deterministic
from temporal_possession.audio import sha
from structured_possession.features import derivative,enrich,event_history,fine_audio,known_dead
from structured_possession.network import FactorizedNet,infer_seed,probabilities,targets,train_seed
from structured_possession.decoder import StructuredParams,decode,objective,transition_kind,refine
from structured_possession.experiment import fit_bundle,predict
from test_temporal_audio import fixture


class StructuredTargetsTests(unittest.TestCase):
    def test_event_types_and_unknown_edges_are_separate(self):
        codes=np.array([3]*10+[0]*10+[1]*10+[2]*10+[0]*10+[3]*10,np.int8)
        y,mask=targets(codes,0)
        self.assertEqual(np.flatnonzero(y[:,0]).tolist(),[30])
        self.assertEqual(np.flatnonzero(y[:,1]).tolist(),[40])
        self.assertEqual(np.flatnonzero(y[:,2]).tolist(),[20])
        self.assertFalse(y[10].any())
        self.assertFalse(y[50].any())
        self.assertFalse(mask[10])
        self.assertFalse(mask[49])

    def test_factorisation_is_normalised_and_ownership_has_no_dead_mass(self):
        p=probabilities(torch.tensor([[0.,20.,-20.]]),torch.tensor([[0.,20.,-20.]]))
        np.testing.assert_allclose(p.sum(dim=1).numpy(),1,atol=1e-6)
        self.assertGreater(p[0,0,1].item(),0.999)
        self.assertGreater(p[0,2,2].item(),0.999)

    def test_history_expires_and_does_not_invent_an_owner(self):
        owner,age=event_history(np.array([0,1,0,0,0,0,0,-1,0]),1,3)
        np.testing.assert_array_equal(owner,[0,1,1,1,1,0,0,-1,-1])
        self.assertTrue(np.isnan(age[0]) and np.isnan(age[5]))
        self.assertEqual(age[8],1)


class StructuredDecoderTests(unittest.TestCase):
    def test_local_refinement_matches_all_ordered_boundary_choices(self):
        rng=np.random.default_rng(5)
        probability=rng.dirichlet(np.ones(3),size=12)
        events=rng.uniform(.1,.9,size=(12,3))
        coarse=np.array([2]*4+[0]*4+[1]*4,np.int8)
        forced=np.r_[True,True,np.zeros(10,bool)]
        params=StructuredParams(refinement_frames=2)
        durations=[(0.,1.),None,(0.,1.)]
        got=refine(probability,events,2,params,durations,forced,coarse)
        expected=-np.inf
        for a in range(2,7):
            for b in range(6,11):
                if a<b:
                    path=np.array([2]*a+[0]*(b-a)+[1]*(12-b))
                    expected=max(expected,objective(path,probability,events,2,params,durations,forced))
        self.assertAlmostEqual(objective(got,probability,events,2,params,durations,forced),expected,places=10)

    def test_many_boundary_neighbourhoods_do_not_build_one_large_graph(self):
        coarse=np.tile(np.repeat([0,1,2],50),100).astype(np.int8)
        probability=np.full((len(coarse),3),.05)
        probability[np.arange(len(coarse)),coarse]=.9
        events=np.full_like(probability,.8)
        params=StructuredParams(refinement_frames=25,max_candidates=100)
        got=refine(probability,events,25,params,None,np.zeros(len(coarse),bool),coarse)
        np.testing.assert_array_equal(got,coarse)

    def test_transition_types_are_source_and_destination_dependent(self):
        self.assertEqual([transition_kind(a,b) for a,b in [(0,2),(1,2),(2,0),(2,1),(0,1),(1,0)]],
                         [0,0,1,1,2,2])

    def test_dense_decoder_matches_exhaustive_paths_with_durations_and_constraints(self):
        rng=np.random.default_rng(4)
        for duration in (None,[(0.,1.),None,(0.,1.)]):
            p=rng.dirichlet(np.ones(3),size=6)
            q=rng.uniform(.05,.95,size=(6,3))
            forced=np.array([True,True,False,False,False,False])
            params=StructuredParams(grid_frames=1,refinement_frames=0,switch_cost=.2)
            got=decode(p,q,2,params,duration,forced)
            expected=max(objective(np.array(path),p,q,2,params,duration,forced)
                         for path in product(range(3),repeat=6))
            self.assertAlmostEqual(objective(got,p,q,2,params,duration,forced),expected,places=10)
            self.assertTrue(np.all(got[:2]==2))

    def test_refinement_never_lowers_the_registered_path_objective(self):
        rng=np.random.default_rng(1)
        p=rng.dirichlet(np.ones(3),size=40)
        q=rng.uniform(.1,.9,size=(40,3))
        coarse=StructuredParams(grid_frames=10,refinement_frames=0,switch_cost=.02)
        fine=StructuredParams(grid_frames=10,refinement_frames=5,switch_cost=.02)
        a,b=decode(p,q,25,coarse),decode(p,q,25,fine)
        self.assertGreaterEqual(objective(b,p,q,25,fine)+1e-10,objective(a,p,q,25,coarse))

    def test_documented_fact_matches_exact_exclusive_endpoint(self):
        cache=FeatureCache(np.zeros((199319,1)),('x',),25,((1,11185,90651),(2,110808,190197)),{})
        mask=known_dead(cache)
        self.assertEqual(np.flatnonzero(mask)[[0,-1]].tolist(),[110808,113741])
        self.assertEqual(mask.sum(),2934)
        self.assertFalse(mask[113742])


class StructuredFeatureTests(unittest.TestCase):
    def test_physics_features_do_not_carry_contact_history_across_halves(self):
        n=100
        halves=((1,0,39),(2,60,99))
        names=('ball_speed','d_home','d_away','whistle')
        cache=FeatureCache(np.zeros((n,4)),names,25,halves,{})
        ball=np.column_stack([np.arange(n)*.01,np.zeros(n)]).astype(np.float32)
        players=np.zeros((n,2,2),np.float32)
        match=SimpleNamespace(n_frames=n,halves=halves,ball=ball,players=players,fps=25,
                              team_by_frame=lambda:np.tile([0,1],(n,1)))
        with patch('possession.signals.contacts',side_effect=[np.r_[np.zeros(39),1],np.zeros(40)]):
            rich=enrich(cache,match)
        j=rich.names.index('contact_last_team')
        self.assertEqual(rich.x[39,j],1)
        self.assertEqual(rich.x[60,j],0)
        self.assertTrue(np.isnan(rich.x[40:60,j]).all())

    def test_fine_audio_matches_a_synthetic_whistle_and_isolates_halves(self):
        rate,fps=32000,25
        t=np.arange(rate*4)/rate
        wave=(.2*np.sin(2*np.pi*3800*t)).astype(np.float32)
        halves=((1,0,29),(2,60,89))
        names,a=fine_audio(wave,halves)
        changed=wave.copy()
        changed[60*1280:]*=0
        _,b=fine_audio(changed,halves)
        np.testing.assert_array_equal(a[:30],b[:30])
        self.assertTrue(np.isnan(a[30:60]).all())
        peak=a[:30,names.index('fine_audio_peak_hz')]
        self.assertTrue(np.all(np.abs(peak-3800)<20))


class StructuredNetworkTests(unittest.TestCase):
    def test_complete_study_including_grouped_folds_and_saved_predictions(self):
        from structured_possession.experiment import compare
        base,audio=fixture(12000)
        rich=FeatureCache(np.column_stack([base.x,np.zeros(len(base.x))]),
                          (*base.names,'extra_physics'),base.fps,base.halves,{})
        train,validation=np.full(12000,'X'),np.full(12000,'X')
        pattern=np.repeat(['H','A','D','H','A'],160)
        train[100:900],train[4100:4900]=pattern,pattern
        validation[10000:10100],validation[10100:10200],validation[10200:10300]='H','A','D'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            bp,rp,ap=root/'base.npz',root/'rich.npz',root/'audio.npz'
            base.save(bp)
            rich.save(rp)
            audio.save(ap)
            tr,va=root/'train.csv',root/'validation.csv'
            write_labels(tr,train)
            write_labels(va,validation)
            audit=root/'audit.json'
            audit.write_text(json.dumps(dict(train_labels_sha256=sha(tr),validation_labels_sha256=sha(va),
                                             annotation_status='synthetic draft')))
            args=Namespace(features=str(bp),rich_features=str(rp),audio=str(ap),train_labels=str(tr),
                validation_labels=str(va),audit=str(audit),out=str(root/'study'),device='cpu',
                epochs=1,seeds=[0],bootstrap=0,compare=[],cross_validate=True)
            compare(args)
            report=json.loads((root/'study/report.json').read_text())
            self.assertEqual(len(report['cv']),2)
            self.assertEqual(report['experiment']['training_frames'],1600)
            self.assertEqual(report['experiment']['validation_frames'],300)
            for name in ('factorized_short_base','factorized_short_rich','factorized_long_rich'):
                model=root/'study'/name
                cp=bp if name.endswith('_base') else rp
                output=root/(name+'.csv')
                with patch('structured_possession.experiment.read_labels',side_effect=AssertionError('Reference read')):
                    predict(Namespace(features=str(cp),audio=str(ap),model=str(model),out=str(output),device='cpu'))
                self.assertEqual(output.read_bytes(),(model/'predictions.csv').read_bytes())
            from structured_possession.calibrate import run as calibrate
            calibration=root/'calibration'
            calibrate(Namespace(source_study=str(root/'study'),features=str(bp),rich_features=str(rp),
                audio=str(ap),train_labels=str(tr),validation_labels=str(va),out=str(calibration),
                device='cpu',bootstrap=0))
            decision=json.loads((calibration/'selected_decoder.json').read_text())
            self.assertFalse(decision['validation_used'])
            for name in ('baseline','structured'):
                chosen=decision['costs'][name]
                grid=decision['cv'][name]
                self.assertEqual(grid[str(chosen)]['spell_f1_mean'],max(v['spell_f1_mean'] for v in grid.values()))
            regenerated=root/'calibrated-structured.csv'
            with patch('structured_possession.experiment.read_labels',side_effect=AssertionError('Reference read')):
                predict(Namespace(features=str(rp),audio=str(ap),model=str(calibration/'models/structured'),
                                  out=str(regenerated),device='cpu'))
            self.assertEqual(regenerated.read_bytes(),(calibration/'structured.csv').read_bytes())

    def test_chunking_matches_whole_half_and_other_half_is_isolated(self):
        deterministic(2)
        config=NetworkConfig(width=4,dilations=(1,2),dropout=0)
        model=FactorizedNet(4,config).eval()
        x=np.random.default_rng(2).normal(size=(270,4)).astype(np.float32)
        halves=((1,10,129),(2,150,249))
        a,q=infer_seed(model,x,halves,[2,3,4],'cpu',core=120,batch_size=1)
        b,r=infer_seed(model,x,halves,[2,3,4],'cpu',core=16,batch_size=3)
        np.testing.assert_allclose(a,b,atol=1e-6)
        np.testing.assert_allclose(q,r,atol=1e-6)
        x[150:]+=1000
        c,_=infer_seed(model,x,halves,[2,3,4],'cpu',core=16,batch_size=3)
        np.testing.assert_array_equal(b[10:130],c[10:130])

    def test_seeded_training_and_prediction_roundtrip_without_labels(self):
        cache,audio=fixture(600)
        codes=np.array([3]*10+[0]*50+[1]*50+[2]*50+[0]*50+[1]*50+[3]*340,np.int8)
        selected=codes!=3
        config=NetworkConfig(width=4,dilations=(1,),epochs=1,core_frames=64)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cp,ap=root/'features.npz',root/'audio.npz'
            cache.save(cp)
            audio.save(ap)
            info=fit_bundle(root/'model',cache,audio,codes,selected,config,[0],'cpu')
            info.update(feature_sha256=sha(cp),audio_sha256=sha(ap))
            (root/'model/model.json').write_text(json.dumps(info))
            args=Namespace(model=str(root/'model'),features=str(cp),audio=str(ap),device='cpu',out=str(root/'one.csv'))
            with patch('structured_possession.experiment.read_labels',side_effect=AssertionError('Labels read at inference')):
                predict(args)
                args.out=str(root/'two.csv')
                predict(args)
            self.assertEqual((root/'one.csv').read_bytes(),(root/'two.csv').read_bytes())


if __name__=='__main__':
    unittest.main()
