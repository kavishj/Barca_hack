import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from external_possession.temporal_target import target_state,extract


class TemporalTargetTests(unittest.TestCase):
    def test_owner_ablation_preserves_dead_and_fixed_boundaries(self):
        gp=np.array([[.2,.3,.5],[.6,.1,.3],[.1,.7,.2]],np.float32)
        gq=np.array([.1,.2,.3]);turn=np.array([.2,0,.1]);observed=np.array([True,False,True])
        old=np.array([.6,.7,.8]);temporal=np.array([.9,.2,.4]);graded=np.ones(3,bool)
        observed_p,observed_q=target_state('temporal_observed',gp,gq,old,turn,observed,temporal,graded)
        all_p,all_q=target_state('temporal_all',gp,gq,old,turn,observed,temporal,graded)
        np.testing.assert_array_equal(observed_p[:,2],gp[:,2]);np.testing.assert_array_equal(all_p[:,2],gp[:,2])
        np.testing.assert_array_equal(observed_p[1],gp[1])
        np.testing.assert_allclose(all_p[1],[.14,.56,.3],atol=1e-7)
        np.testing.assert_array_equal(observed_q,all_q);np.testing.assert_array_equal(all_q,np.maximum(gq,turn))
        fixed,_=target_state('fixed',gp,gq,old,turn,observed,temporal,graded)
        np.testing.assert_allclose(fixed[0],[.3,.2,.5],atol=1e-7)

    def test_failed_external_gate_stops_before_target_observation_access(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'report.json').write_text(json.dumps(dict(gate_passed=False)))
            (root/'experiment.json').write_text('{}')
            args=SimpleNamespace(out=str(root/'output'),model=str(root))
            with self.assertRaisesRegex(ValueError,'gate'):
                extract(args)
            self.assertFalse((root/'output').exists())


if __name__=='__main__':unittest.main()
