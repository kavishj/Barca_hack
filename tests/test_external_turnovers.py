from itertools import product
import copy
import unittest
import numpy as np
from external_possession.turnovers import decode,fees,training_prior,confidence_turnovers
from learned_possession.segment import decode as fixed_decode,DecoderParams
from learned_possession.features import FeatureCache
from test_relational_possession import graph_fixture


class TypedTurnoverTests(unittest.TestCase):
    def test_dynamic_programme_matches_all_possible_typed_paths(self):
        rng=np.random.default_rng(13)
        for policy in ('typed_fixed','typed_adaptive','typed_gated'):
            for n in (3,5,7):
                p=rng.dirichlet([1,1,1],size=n);g=rng.uniform(.01,.99,n);t=rng.uniform(.01,.99,n)
                cost=fees(g,t,.02,policy);path=decode(p,g,t,1,.02,policy,grid_frames=1)
                def score(x):return float(np.log(p[np.arange(n),x]).sum()-sum(cost[i,x[i-1],x[i]] for i in range(1,n) if x[i]!=x[i-1]))
                best=max(score(np.array(x)) for x in product(range(3),repeat=n))
                self.assertAlmostEqual(score(path),best,places=12)

    def test_typed_fixed_agrees_with_original_when_all_boundary_scores_agree(self):
        rng=np.random.default_rng(14);p=rng.dirichlet([1,1,1],size=30);q=rng.uniform(.01,.99,30)
        a=fixed_decode(p,q,1,DecoderParams(switch_cost=8,grid_frames=1))
        b=decode(p,q,q,1,.02,'typed_fixed',grid_frames=1)
        np.testing.assert_array_equal(a,b)

    def test_subsecond_turnover_is_recovered_without_lowering_dead_transition_fees(self):
        p=np.tile([.01,.989,.001],(200,1));p[90:109]=[.989,.01,.001]
        g=np.ones(200)*.01;t=np.zeros(200);t[[90,109]]=.9
        a=decode(p,g,t,25,.01,'typed_adaptive',grid_frames=1)
        self.assertTrue(np.all(a[90:109]==0) and np.all(a[:90]==1) and np.all(a[109:]==1))
        p[90:109]=[.001,.01,.989]
        self.assertTrue(np.all(decode(p,g,t,25,.01,'typed_adaptive',grid_frames=1)==1))
        constant=np.tile([.9,.05,.05],(100,1))
        self.assertTrue(np.all(decode(constant,np.ones(100),np.ones(100),25,.01,'typed_adaptive',grid_frames=1)==0))

    def test_prior_ignores_unknown_edges_and_gating_removes_unobserved_velocity(self):
        codes=np.full(300,3,np.int8);codes[30:230]=np.repeat([0,1,2,0],50)
        p,info=training_prior(codes);self.assertGreater(p,0);self.assertLess(p,1);self.assertEqual(info['positive_band_frames'],7)
        graph=graph_fixture(500,((1,0,199),(2,300,499)))
        cache=FeatureCache(np.ones((500,1)),('ball_visible',),25,graph.halves,{})
        owner=np.where(np.arange(500)<100,.99,.01).astype(np.float32)
        q=confidence_turnovers(graph,owner,cache)
        self.assertGreater(q.max(),0)
        alternate=confidence_turnovers(graph,1-owner,cache);np.testing.assert_allclose(q,alternate,atol=1e-7)
        other=copy.deepcopy(graph);other.nodes[:,-1,13]=0
        self.assertTrue(np.all(confidence_turnovers(other,owner,cache)==0))
        with self.assertRaises(ValueError):decode(np.ones((3,3))/3,np.zeros(3),np.zeros(3),25,0,'typed_adaptive')


if __name__=='__main__':unittest.main()
