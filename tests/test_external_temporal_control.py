import unittest
import numpy as np
import torch
from external_possession.temporal_control import (
    ControlNet, ball_mask, boundary_targets, block_context, random_blocks, infer, external_scores)


class ExternalTemporalControlTests(unittest.TestCase):
    def test_shared_owner_and_invariant_boundary(self):
        torch.manual_seed(3);torch.set_num_threads(2)
        model=ControlNet().eval();x=torch.randn(2,2,35,75);point=torch.randn(2,2,75)
        owner,q=model(x,point)
        torch.testing.assert_close(owner,point)
        torch.nn.init.normal_(model.owner.weight)
        owner,q=model(x,point);swapped,sq=model(x.flip(1),point.flip(1))
        torch.testing.assert_close(owner.flip(1),swapped);torch.testing.assert_close(q,sq)

    def test_ball_mask_preserves_player_kinematics_and_source(self):
        rng=np.random.default_rng(2)
        d=dict(nodes=rng.normal(size=(2,47,15)).astype(np.float32),roles=np.zeros((2,47),np.int8),
            valid=np.ones((2,47),bool),positioned=np.ones((2,47),bool),confidence=np.ones(2))
        original=d['nodes'].copy();masked=ball_mask(d)
        np.testing.assert_array_equal(d['nodes'],original)
        np.testing.assert_array_equal(masked['nodes'][:,:46,:5],original[:,:46,:5])
        self.assertFalse(masked['nodes'][:,:,5:10].any());self.assertFalse(masked['nodes'][:,:,14].any())
        self.assertFalse(masked['nodes'][:,-1,:10].any());self.assertFalse(masked['positioned'][:,-1].any())

    def test_boundaries_never_bridge_unknown_or_half(self):
        labels=np.array([0,0,1,1,2,0,0,1,1,0,0,0],np.int8)
        half=np.array([1]*9+[2]*3)
        y,m=boundary_targets(labels,half)
        np.testing.assert_array_equal(np.flatnonzero(m),[1,2,6,7,10])
        self.assertTrue(y[2] and y[7]);self.assertFalse(y[9])
        x=np.ones((12,2,35),np.float32);x[9:]=9
        context=block_context(x,7,4,2,0,9)
        self.assertTrue((context[:4]==1).all());self.assertFalse(context[4:].any())

    def test_mask_blocks_independent_and_expected_coverage(self):
        halves=np.r_[np.ones(50000),np.full(50000,2)]
        mask=random_blocks(halves,np.random.default_rng(4))
        self.assertTrue(.17<float(mask.mean())<.24)
        np.testing.assert_array_equal(mask,random_blocks(halves,np.random.default_rng(4)))

    def test_chunked_inference_has_same_local_receptive_field(self):
        torch.manual_seed(7);torch.set_num_threads(2);model=ControlNet().eval()
        torch.nn.init.normal_(model.owner.weight)
        rng=np.random.default_rng(4);x=rng.normal(size=(1100,2,35)).astype(np.float32)
        point=rng.normal(size=(1100,2)).astype(np.float32)
        logits,q=infer(model,x,point,np.ones(1100),np.zeros(35),np.ones(35),'cpu')
        padded=block_context(x,0,1100,30,0,1100).transpose(1,2,0)[None]
        pp=block_context(point,0,1100,30,0,1100).T[None]
        with torch.inference_mode():expected,eq=model(torch.from_numpy(padded),torch.from_numpy(pp))
        np.testing.assert_allclose(logits,expected[0,:,30:1130].T.numpy(),atol=2e-5,rtol=1e-5)
        np.testing.assert_allclose(q,eq[0,30:1130].numpy(),atol=2e-5,rtol=1e-5)

    def test_conditional_metrics_do_not_use_reference_to_predict(self):
        full=np.array([3,0,0,0,1,1,2,0,0,0,0],np.int8)
        data=dict(full_labels=full,frames=np.arange(1,11),half_ids=np.ones(10),metadata=dict(halves=[(1,1,10)]))
        logits=np.tile([2.,-2.],(10,1));logits[3:5]=[-2,2]
        score,p=external_scores(data,logits)
        self.assertEqual(score['n_gt_turnovers'],1);self.assertEqual(score['typed_turnover_f1_05s'],1.)
        data['full_labels']=np.where(full<2,1-full,full)
        _,p2=external_scores(data,logits);np.testing.assert_array_equal(p,p2)
        self.assertFalse(score['target_match_score']);self.assertTrue(score['conditional_metrics'])


if __name__=='__main__':unittest.main()
