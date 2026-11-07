import unittest
import numpy as np
from boosted_possession.dead_expert import factorized,POLICIES


class DeadExpertTests(unittest.TestCase):
    def test_tabular_owner_is_ignored_and_dead_evidence_is_combined(self):
        old=(np.array([[.2,.3,.5],[.6,.3,.1]],np.float32),np.array([.1,.2]))
        clap=(np.array([[.7,.2,.1],[.1,.3,.6]],np.float32),np.array([.3,.8]))
        trees=(np.array([[.1,.1,.8],[.2,.6,.2]],np.float32),np.array([.5,.4]))
        owner=np.array([.8,.7]);turn=np.array([.2,0]);seen=np.array([True,False])
        p,q=factorized('trees_dead',old,clap,trees,owner,turn,seen)
        np.testing.assert_allclose(p,[[.16,.04,.8],[.8*2/3,.8/3,.2]],atol=1e-7)
        changed=(trees[0][:,[1,0,2]],trees[1])
        for policy in POLICIES[1:]:
            p,q=factorized(policy,old,clap,trees,owner,turn,seen)
            other,oq=factorized(policy,old,clap,changed,owner,turn,seen)
            np.testing.assert_array_equal(p,other);np.testing.assert_array_equal(q,oq)
            np.testing.assert_allclose(p.sum(1),1,atol=1e-7)
        mean,mq=factorized('trees_graph_mean',old,clap,trees,owner,turn,seen)
        np.testing.assert_allclose(mean[:,2],[.65,.15],atol=1e-7)
        np.testing.assert_allclose(mq,[.3,.3],atol=1e-7)

    def test_independent_owner_survives_old_dead_probability_saturation(self):
        old=(np.array([[0.,0.,1.]],np.float32),np.zeros(1))
        alternative=(np.array([[.1,.7,.2]],np.float32),np.zeros(1))
        p,_=factorized('trees_dead',old,alternative,alternative,np.array([.9]),np.zeros(1),np.ones(1,bool))
        np.testing.assert_allclose(p,[[.72,.08,.2]],atol=1e-7)


if __name__=='__main__':unittest.main()
