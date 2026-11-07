import unittest
import numpy as np
import torch
from external_possession.target_probe import fit,probability,representations,probe_features
from external_possession.temporal_control import infer
from external_possession.dead_control import DeadControlNet


class TargetProbeTests(unittest.TestCase):
    def test_excluded_windows_cannot_affect_scalers_or_coefficients(self):
        rng=np.random.default_rng(9);owner=rng.normal(size=(90,33));dead=rng.normal(size=(90,67)).astype(np.float32)
        labels=np.tile(np.arange(3,dtype=np.int8),30);selected=np.arange(90)<60
        a,stats=fit(owner,dead,labels,selected)
        changed_owner=owner.copy();changed_dead=dead.copy();changed_labels=labels.copy()
        changed_owner[~selected]=999;changed_dead[~selected]=-999;changed_labels[~selected]=3
        b,other=fit(changed_owner,changed_dead,changed_labels,selected)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(stats,other);self.assertEqual(stats['known_samples'],60)
        self.assertEqual(stats['live_samples'],40)
        p=probability(owner,dead,a);swapped=probability(-owner,dead,a)
        np.testing.assert_allclose(p[:,[1,0,2]],swapped,atol=1e-12)
        np.testing.assert_allclose(p.sum(1),1,atol=1e-12)

    def test_frozen_representations_reproduce_original_heads_and_reset_halves(self):
        torch.manual_seed(7);torch.set_num_threads(2);model=DeadControlNet().eval()
        torch.nn.init.normal_(model.owner.weight);torch.nn.init.normal_(model.dead.weight)
        rng=np.random.default_rng(3);x=rng.normal(size=(24,2,35)).astype(np.float32)
        x[:,1,33:35]=x[:,0,33:35];point=rng.normal(size=(24,2)).astype(np.float32)
        halves=np.r_[np.ones(12),np.full(12,2)];mean=np.zeros(35);scale=np.ones(35)
        h=representations(model,x,halves,mean,scale,'cpu')
        owner,dead=probe_features(h,x,point)
        swapped_owner,swapped_dead=probe_features(h[:,::-1],x[:,::-1],point[:,::-1])
        np.testing.assert_array_equal(-owner,swapped_owner);np.testing.assert_array_equal(dead,swapped_dead)
        changed=x.copy();changed[12:]=8
        other=representations(model,changed,halves,mean,scale,'cpu')
        np.testing.assert_array_equal(h[:12],other[:12])
        logits,_=infer(model,x,point,halves,mean,scale,'cpu')
        hh=torch.from_numpy(h.transpose(1,2,0))
        with torch.inference_mode():
            own=model.owner(hh)[:,0].T+torch.from_numpy(point)
            invariant=torch.cat([hh.mean(0,keepdim=True),(hh[0:1]-hh[1:2]).abs()],dim=1)
            d=model.dead(invariant)[0,0]+own.logsumexp(1)
        expected=torch.cat([own,d[:,None]],dim=1).numpy()
        np.testing.assert_allclose(logits,expected,atol=2e-5,rtol=1e-5)


if __name__=='__main__':unittest.main()
