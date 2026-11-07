import unittest
import numpy as np
from external_possession.visibility import fit,apply


class VisibilityCalibrationTests(unittest.TestCase):
    def test_training_ratios_ignore_unknowns_and_are_team_invariant(self):
        codes=np.array([0]*40+[1]*40+[2]*80+[3]*40,np.int8)
        visible=np.r_[np.ones(70,bool),np.zeros(90,bool),np.ones(40,bool)]
        ratios,info=fit(codes,visible)
        self.assertGreater(ratios[0],1);self.assertLess(ratios[1],1)
        flipped=visible.copy();flipped[codes==3]=~flipped[codes==3]
        a,_=fit(codes,flipped);np.testing.assert_array_equal(a,ratios)
        renamed=np.where(codes<2,1-codes,codes);b,_=fit(renamed,visible);np.testing.assert_array_equal(b,ratios)

    def test_correction_preserves_conditional_owner_and_zero_is_exact_control(self):
        p=np.tile(np.array([.3,.2,.5],np.float32),(4,1));v=np.array([False,True,False,True])
        np.testing.assert_array_equal(apply(p,v,[4,.25],0),p)
        out=apply(p,v,[4,.25],1)
        np.testing.assert_allclose(out[:,0]/(out[:,0]+out[:,1]),.6,atol=1e-7)
        self.assertTrue(np.all(out[~v,2]<.5) and np.all(out[v,2]>.5))
        np.testing.assert_allclose(out.sum(axis=1),1,atol=1e-7)
        np.testing.assert_allclose(apply(p[:,[1,0,2]],v,[4,.25],1),out[:,[1,0,2]],atol=1e-7)

    def test_extreme_posteriors_remain_finite_and_invalid_inputs_are_rejected(self):
        p=np.array([[0,0,1],[.5,.5,0]],np.float32)
        out=apply(p,np.array([False,True]),[4,.25],1)
        self.assertTrue(np.isfinite(out).all());np.testing.assert_allclose(out.sum(axis=1),1,atol=1e-7)
        with self.assertRaises(ValueError):apply(p,np.array([False,True]),[0,1],1)
        with self.assertRaises(ValueError):fit(np.array([3,3]),np.array([False,True]))


if __name__=='__main__':unittest.main()
