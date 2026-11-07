import unittest
import numpy as np
from temporal_possession.audio_blend import POLICIES,blend


class AudioBlendTests(unittest.TestCase):
    def fixture(self):
        old=(np.array([[.2,.3,.5],[.6,.3,.1]],np.float32),np.array([.1,.2]))
        new=(np.array([[.7,.2,.1],[.1,.3,.6]],np.float32),np.array([.3,.8]))
        return old,new,np.array([.8,.7]),np.array([.2,0]),np.array([True,False])

    def test_dead_only_fusion_preserves_missing_ball_owner_and_averages_boundaries(self):
        args=self.fixture()
        p,q=blend('dead_mean',*args)
        np.testing.assert_allclose(p[1],[.65*2/3,.65/3,.35],atol=1e-7)
        np.testing.assert_allclose(q,[.2,.5],atol=1e-7)
        full,_=blend('state_mean',*args);np.testing.assert_allclose(full[1],[.35,.3,.35],atol=1e-7)
        replacement,rq=blend('dead_clap',*args)
        np.testing.assert_allclose(replacement[1],[.4*2/3,.4/3,.6],atol=1e-7)
        np.testing.assert_allclose(rq,[.3,.8],atol=1e-7)

    def test_all_policies_preserve_team_equivariance_and_normalized_mass(self):
        old,new,owner,turn,available=self.fixture()
        for policy in POLICIES:
            p,q=blend(policy,old,new,owner,turn,available)
            swapped,sq=blend(policy,(old[0][:,[1,0,2]],old[1]),(new[0][:,[1,0,2]],new[1]),1-owner,turn,available)
            np.testing.assert_allclose(p[:,[1,0,2]],swapped,atol=1e-7)
            np.testing.assert_allclose(p.sum(1),1,atol=1e-7);np.testing.assert_array_equal(q,sq)


if __name__=='__main__':unittest.main()
