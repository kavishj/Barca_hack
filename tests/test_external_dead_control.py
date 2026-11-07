import unittest
import numpy as np
import torch
from scipy.special import softmax
from external_possession.dead_control import (
    DeadControlNet,baseline_fit,baseline_probability,invariant_features,boundary_targets)
from external_possession.temporal_control import infer


class ExternalDeadControlTests(unittest.TestCase):
    def test_factorized_dead_prior_and_team_symmetry(self):
        torch.manual_seed(2);torch.set_num_threads(2)
        model=DeadControlNet().eval();torch.nn.init.constant_(model.dead.bias,np.log(.4/.6))
        x=torch.randn(1,2,35,70);point=torch.randn(1,2,70)
        logits,q=model(x,point);p=logits.softmax(1)
        torch.testing.assert_close(p[:,2],torch.full_like(p[:,2],.4))
        torch.testing.assert_close(p[:,0]/p[:,:2].sum(1),point.softmax(1)[:,0])
        torch.nn.init.normal_(model.dead.weight);torch.nn.init.normal_(model.owner.weight)
        logits,q=model(x,point);swapped,sq=model(x.flip(1),point.flip(1))
        torch.testing.assert_close(logits[:,[1,0,2]],swapped);torch.testing.assert_close(q,sq)

    def test_external_baseline_ignores_unknown_and_is_invariant(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(80,2,35)).astype(np.float32)
        # Visibility/confidence are shared, as in the real frozen feature schema.
        x[:,1,33:35]=x[:,0,33:35];point=rng.normal(size=(80,2)).astype(np.float32)
        labels=np.tile(np.arange(4,dtype=np.int8),20);fit=baseline_fit(x,point,labels)
        changed=x.copy();changed[labels==3]=999;changed_point=point.copy();changed_point[labels==3]=-999
        other=baseline_fit(changed,changed_point,labels)
        for key in fit:np.testing.assert_array_equal(fit[key],other[key])
        p=baseline_probability(x,point,fit);swapped=baseline_probability(x[:,::-1],point[:,::-1],fit)
        np.testing.assert_allclose(p[:,[1,0,2]],swapped,atol=1e-7)
        self.assertEqual(invariant_features(x,point).shape,(80,67))

    def test_dead_transitions_are_supervised_without_bridging_unknown(self):
        labels=np.array([0,0,2,2,3,1,1,2,2,0,0,0],np.int8);halves=np.array([1]*9+[2]*3)
        y,mask=boundary_targets(labels,halves)
        self.assertTrue(y[2] and y[7]);self.assertFalse(y[9])
        np.testing.assert_array_equal(np.flatnonzero(mask),[1,2,6,7,10])

    def test_generic_chunked_inference_preserves_three_states(self):
        torch.manual_seed(4);torch.set_num_threads(2);model=DeadControlNet().eval()
        x=np.ones((9,2,35),np.float32);point=np.zeros((9,2),np.float32)
        logits,q=infer(model,x,point,np.r_[np.ones(4),np.full(5,2)],np.zeros(35),np.ones(35),'cpu')
        self.assertEqual(logits.shape,(9,3));self.assertEqual(q.shape,(9,))
        np.testing.assert_allclose(softmax(logits,axis=1)[:,2],.5,atol=1e-7)


if __name__=='__main__':unittest.main()
