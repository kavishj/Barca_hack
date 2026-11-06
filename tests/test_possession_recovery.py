from itertools import product
from argparse import Namespace
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from learned_possession.segment import decode as fixed_decode,DecoderParams
from learned_possession.features import FeatureCache
from relational_possession.model import RelationalNet,Config
from relational_possession.engine import tensors,train_seed as old_train
from temporal_possession.network import deterministic
from possession_recovery.decoder import decode
from possession_recovery.loss import spell_weights
from possession_recovery.features import enrich
from possession_recovery.model import RecoveryNet
from possession_recovery.engine import train_seed
from test_relational_possession import graph_fixture


class RecoveryDecoderTests(unittest.TestCase):
    def test_exact_objective_matches_exhaustive_state_sequences(self):
        rng=np.random.default_rng(42)
        for length in (3,5,7):
            p=rng.dirichlet([1,1,1],size=length);q=rng.uniform(0,1,size=length);prior=.03;fee=2.
            actual=decode(p,q,25,prior,switch_cost=fee,grid_frames=1)
            def score(path):
                return float(np.log(p[np.arange(length),path]).sum()/25-sum(
                    fee*prior/(q[t]+prior) for t in range(1,length) if path[t]!=path[t-1]))
            optimal=max(score(np.array(x)) for x in product(range(3),repeat=length))
            self.assertAlmostEqual(score(actual),optimal,places=12)

    def test_strong_boundary_can_recover_subsecond_spell_without_transition_rewards(self):
        p=np.tile([.01,.989,.001],(200,1));p[90:109]=[.989,.01,.001]
        q=np.zeros(200);q[[90,109]]=.8
        old=fixed_decode(p,q,25,DecoderParams(grid_frames=1,switch_cost=8),None)
        new=decode(p,q,25,.015,grid_frames=1)
        self.assertTrue(np.all(old==1))
        np.testing.assert_array_equal(new[90:109],np.zeros(19,int))
        self.assertTrue(np.all(new[:90]==1) and np.all(new[109:]==1))
        constant=np.tile([.9,.05,.05],(200,1))
        self.assertTrue(np.all(decode(constant,np.ones(200),25,.015,grid_frames=1)==0))

    def test_invalid_probabilities_and_training_prior_are_rejected(self):
        p=np.ones((10,3))/3
        with self.assertRaises(ValueError):decode(p,np.zeros(10),25,0)
        with self.assertRaises(ValueError):decode(p,np.full(10,np.nan),25,.01)


class RecoveryLossTests(unittest.TestCase):
    def test_censored_edges_keep_unit_weight_and_live_dead_mass_is_preserved(self):
        codes=np.array([3]*3+[0]*15+[1]*4+[0]*40+[2]*20+[0]*5+[3]*3,np.int8)
        w,info=spell_weights(codes)
        self.assertEqual(info['complete_live_spells'],2)
        np.testing.assert_array_equal(w[3:18],np.ones(15))
        np.testing.assert_array_equal(w[82:87],np.ones(5))
        self.assertGreater(w[18],w[22])
        self.assertAlmostEqual(float(w[codes<2].sum()),int((codes<2).sum()),places=4)
        self.assertTrue(np.all(w[codes==2]==1) and np.all(w[codes==3]==0))


class RecoveryGraphTests(unittest.TestCase):
    def fixture(self):
        graph=graph_fixture(500,((1,0,199),(2,300,499)))
        names=('contact_last_team','contact_age_s','reception_last_team','reception_age_s','d_home','d_away','ball_speed','ball_visible')
        x=np.zeros((500,len(names)),np.float32);x[:,0]=1;x[:,1]=2;x[:,2]=-1;x[:,3]=3
        x[:,4]=1;x[:,5]=4;x[:,6]=2;x[:,7]=1
        return graph,FeatureCache(x,names,25,graph.halves,{})

    def test_history_is_team_equivariant_and_base_observations_stay_exact(self):
        graph,cache=self.fixture();new=enrich(graph,cache)
        np.testing.assert_array_equal(new.nodes[:,:,:15],graph.nodes)
        other=copy.deepcopy(graph);other.roles=np.where(other.roles<2,1-other.roles,other.roles)
        flipped=copy.deepcopy(cache);flipped.x[:,[0,2]]*=-1;flipped.x[:,[4,5]]=flipped.x[:,[5,4]]
        alternate=enrich(other,flipped)
        np.testing.assert_array_equal(new.nodes,alternate.nodes)
        self.assertTrue(np.isfinite(new.nodes).all())

    def test_disabled_history_and_natural_loss_reproduce_original_optimization(self):
        graph,cache=self.fixture();augmented=enrich(graph,cache)
        rng=np.random.default_rng(1);gx=rng.normal(size=(500,6)).astype(np.float32);px=rng.normal(size=(500,2,4)).astype(np.float32)
        codes=np.full(500,3,np.int8);codes[10:190]=np.repeat([0,1,2],60);selected=codes!=3
        config=Config(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=.15,warm_seconds=1,epochs=2)
        old,_=old_train(gx,px,graph,codes,selected,config,'graph_tcn',3,'cpu')
        new,_=train_seed(gx,px,augmented,codes,selected,config,3,'cpu',history=False,balanced=False)
        for name,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[name],atol=0,rtol=0)

    def test_history_network_output_remains_team_equivariant(self):
        graph,cache=self.fixture();graph=enrich(graph,cache);deterministic(2)
        config=Config(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=0)
        model=RecoveryNet(6,4,27,config).eval()
        with torch.no_grad():model.spatial.input.history.weight.normal_()
        data=tensors(graph,0,150,1,'cpu');gx=torch.randn(150,6);px=torch.randn(150,2,4)
        flipped={key:value.clone() for key,value in data.items()}
        flipped['roles']=torch.where(flipped['roles']<2,1-flipped['roles'],flipped['roles'])
        with torch.inference_mode():a,b,_=model(gx,px,data);c,d,_=model(gx,px[:,[1,0]],flipped)
        torch.testing.assert_close(a[:,[1,0,2]],c,atol=3e-6,rtol=2e-5);torch.testing.assert_close(b,d,atol=3e-6,rtol=2e-5)

    def test_complete_study_selects_before_main_scoring_and_deploys_without_labels(self):
        from test_temporal_audio import fixture
        from learned_possession.__main__ import write_labels
        from temporal_possession.audio import sha
        from temporal_possession.experiment import metrics
        from relational_possession.experiment import compare as baseline_compare
        from possession_recovery.features import prepare
        from possession_recovery.experiment import compare,predict
        base,audio=fixture(10000)
        base=FeatureCache(np.column_stack([base.x,np.ones((10000,2))]),
            (*base.names,'home_nearest_1','away_nearest_1'),25,base.halves,{})
        graph=graph_fixture(10000,base.halves)
        names=('contact_last_team','contact_age_s','reception_last_team','reception_age_s','d_home','d_away','ball_speed','ball_visible')
        x=np.tile([1,2,-1,3,1,4,2,1],(10000,1)).astype(np.float32)
        rich=FeatureCache(x,names,25,base.halves,{})
        train=np.full(10000,'X');validation=np.full(10000,'X')
        train[100:300]=np.repeat(['H','A','D','H'],50);train[3500:3700]=train[100:300]
        validation[8000:8150]=np.repeat(['H','A','D'],50)
        def small(**kwargs):
            values=dict(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=0,warm_seconds=1)
            values.update(kwargs);return Config(**values)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            fp,ap,gp,rp,hp=(root/name for name in ('features.npz','audio.npz','graph.npz','rich.npz','history.npz'))
            base.save(fp);audio.save(ap);graph.save(gp);rich.save(rp)
            prepare(Namespace(graph=str(gp),rich_features=str(rp),out=str(hp)))
            tr,va=root/'train.csv',root/'validation.csv';write_labels(tr,train);write_labels(va,validation)
            audit=root/'audit.json';audit.write_text(json.dumps(dict(train_labels_sha256=sha(tr),validation_labels_sha256=sha(va))))
            source=root/'control'
            with patch('relational_possession.experiment.Config',side_effect=small):
                baseline_compare(Namespace(features=str(fp),audio=str(ap),graph=str(gp),train_labels=str(tr),
                    validation_labels=str(va),audit=str(audit),out=str(source),epochs=1,seeds=[0],
                    device='cpu',bootstrap=0,compare=[],cross_validate=True,calibrate=False))
            study=root/'recovery';main_calls=[]
            def guarded(truth,predictions,bootstrap=0):
                is_main=np.array_equal(truth,np.array([{'H':0,'A':1,'D':2,'X':3}[v] for v in validation],np.int8))
                if is_main:
                    self.assertTrue((study/'decision.json').exists());main_calls.append(1)
                return metrics(truth,predictions,bootstrap)
            with patch('possession_recovery.experiment.Config',side_effect=small),patch('possession_recovery.experiment.metrics',side_effect=guarded):
                compare(Namespace(features=str(fp),audio=str(ap),graph=str(hp),train_labels=str(tr),validation_labels=str(va),
                    source_study=str(source),out=str(study),epochs=1,seeds=[0],device='cpu',bootstrap=0))
            self.assertEqual(len(main_calls),2)
            report=json.loads((study/'report.json').read_text());self.assertEqual(len(report['oof_scores']),8)
            output=root/'regenerated.csv'
            with patch('possession_recovery.experiment.read_labels',side_effect=AssertionError('Reference read')):
                predict(Namespace(features=str(fp),audio=str(ap),graph=str(hp),model=str(study/'deployment/model'),
                    out=str(output),device='cpu'))
            self.assertEqual(output.read_bytes(),(study/'selected_predictions.csv').read_bytes())


if __name__=='__main__':unittest.main()
