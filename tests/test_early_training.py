import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from relational_possession.early import source_mask,fit_group


class EarlyTrainingTests(unittest.TestCase):
    def test_preprocessing_mask_must_match_held_out_fold_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d);selected=np.array([True,False,True,False])
            np.savez(source/'training_mask.npz',selected=selected)
            source_mask(source,selected.copy())
            changed=selected.copy();changed[1]=True
            with self.assertRaisesRegex(ValueError,'mask differs'):source_mask(source,changed)

    def test_tampered_source_scaler_stops_before_training_or_output(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';(source/'graph_tcn').mkdir(parents=True);(source/'preprocessing').mkdir()
            selected=np.array([True,False,True,False]);np.savez(source/'training_mask.npz',selected=selected)
            (source/'preprocessing/symmetric.npz').write_bytes(b'changed')
            (source/'graph_tcn/model.json').write_text(json.dumps(dict(config=dict(epochs=60),seeds=[0,1,2],
                normalizer_hashes={'symmetric.npz':'wrong_sha256'})))
            out=Path(d)/'output'
            with self.assertRaisesRegex(ValueError,'preprocessing changed'):
                fit_group(out,source,None,None,None,None,selected,'cpu')
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
