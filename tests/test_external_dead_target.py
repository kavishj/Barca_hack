from types import SimpleNamespace
import unittest
import numpy as np
from external_possession.dead_target import interpolate_probabilities,blend_state


class ExternalDeadTargetTests(unittest.TestCase):
    def test_interpolation_resets_at_halves_and_preserves_probabilities(self):
        halves=((1,2,8),(2,12,18))
        graph=SimpleNamespace(frames=np.array([2,5,8,12,15,18]),half_ids=np.array([1,1,1,2,2,2]),halves=halves,fps=25)
        base=SimpleNamespace(x=np.zeros((20,1)),halves=halves,fps=25)
        coarse=np.array([[.9,.1,0],[.3,.7,0],[.1,.8,.1],[.05,.05,.9],[.05,.05,.9],[.05,.05,.9]])
        fine=interpolate_probabilities(graph,coarse,base)
        np.testing.assert_allclose(fine.sum(1),1,atol=1e-7)
        np.testing.assert_allclose(fine[3],[.7,.3,0],atol=1e-7)
        np.testing.assert_allclose(fine[12],[.05,.05,.9],atol=1e-7)
        np.testing.assert_allclose(fine[9:12],np.full((3,3),1/3),atol=1e-7)
        with self.assertRaises(ValueError):interpolate_probabilities(graph,coarse*2,base)

    def test_full_blend_changes_dead_only_on_graded_frames(self):
        gp=np.array([[.3,.2,.5],[.6,.1,.3]],np.float32);gq=np.array([.2,.1]);turn=np.array([.1,0])
        external=np.array([[.1,.1,.8],[.1,.2,.7]],np.float32);observed=np.array([True,False]);graded=np.array([True,False])
        p,q=blend_state('external_0.5',gp,gq,np.array([.6,.8]),turn,observed,external,graded)
        np.testing.assert_allclose(p[0],[.2,.15,.65],atol=1e-7);np.testing.assert_array_equal(p[1],gp[1])
        np.testing.assert_array_equal(q,np.maximum(gq,turn))
        control,_=blend_state('fixed',gp,gq,np.array([.6,.8]),turn,observed,external,graded)
        typed,_=blend_state('typed_gated',gp,gq,np.array([.6,.8]),turn,observed,external,graded)
        np.testing.assert_array_equal(control,typed)


if __name__=='__main__':unittest.main()
