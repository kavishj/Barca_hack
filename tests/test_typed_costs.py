from itertools import product
import unittest
import numpy as np
from learned_possession.segment import decode,DecoderParams
from physical_possession.typed import matrix


class TypedCostTests(unittest.TestCase):
    def test_dense_decoder_matches_exhaustive_typed_transition_objective(self):
        rng=np.random.default_rng(4)
        for n in (3,5,7):
            p=rng.dirichlet([1,1,1],size=n);q=rng.uniform(.01,.99,size=n);cost=matrix(.3,.8)
            params=DecoderParams(grid_frames=1,switch_cost=0,duration_weight=0)
            result=decode(p,q,1,params,transition_costs=cost)
            def score(path):return float(np.log(p[np.arange(n),path]).sum()-sum(
                cost[path[t-1],path[t]]-.5*np.log(q[t]) for t in range(1,n) if path[t]!=path[t-1]))
            optimum=max(score(np.array(x)) for x in product(range(3),repeat=n))
            self.assertAlmostEqual(score(result),optimum,places=12)

    def test_uniform_cost_reproduces_original_and_invalid_matrix_is_rejected(self):
        rng=np.random.default_rng(5);p=rng.dirichlet([1,1,1],size=30);q=rng.uniform(0,1,30)
        params=DecoderParams(grid_frames=1,switch_cost=.2,duration_weight=0)
        np.testing.assert_array_equal(decode(p,q,1,params),decode(p,q,1,params,transition_costs=matrix(.2,.2)))
        with self.assertRaises(ValueError):decode(p,q,1,params,transition_costs=np.ones((3,3)))
        with self.assertRaises(ValueError):decode(p,q,1,params,transition_costs=matrix(-1,2))

    def test_brief_turnover_is_recoverable_while_dead_switches_keep_their_cost(self):
        p=np.tile([.01,.989,.001],(200,1));p[90:109]=[.989,.01,.001];q=np.ones(200)*.8
        params=DecoderParams(grid_frames=1,switch_cost=8,duration_weight=0)
        old=decode(p,q,25,params);new=decode(p,q,25,params,transition_costs=matrix(1,8))
        self.assertTrue(np.all(old==1));self.assertTrue(np.all(new[90:109]==0))
        p[90:109]=[.001,.01,.989]
        self.assertTrue(np.all(decode(p,q,25,params,transition_costs=matrix(1,8))==1))


if __name__=='__main__':unittest.main()
