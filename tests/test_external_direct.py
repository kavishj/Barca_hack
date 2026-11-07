import unittest
import numpy as np
from scipy.special import expit
from external_possession.direct import probe_fit,fine_scores
from learned_possession.features import FeatureCache
from test_relational_possession import graph_fixture
from physical_possession.model import mix


class DirectExternalTests(unittest.TestCase):
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
