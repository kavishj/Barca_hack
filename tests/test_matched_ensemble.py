from argparse import Namespace
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from learned_possession.__main__ import write_labels
from temporal_possession.audio import sha
from temporal_possession.experiment import metrics
from relational_possession.model import Config
from test_temporal_audio import fixture
from test_relational_possession import graph_fixture


class MatchedEnsembleTests(unittest.TestCase):
    def test_matched_fold_ensembles_select_before_main_and_do_not_refit_global_weights(self):
        from relational_possession.experiment import compare as original_compare
        from relational_possession.matched import compare,predict
        base,audio=fixture(10000);graph=graph_fixture(10000,base.halves)
        train=np.full(10000,'X');validation=train.copy()
        train[100:400]=np.repeat(['H','A','D'],100);train[3500:3800]=train[100:400];validation[8000:8300]=train[100:400]
        truth=np.array([{'H':0,'A':1,'D':2,'X':3}[s] for s in validation],np.int8)
        def small(**kwargs):
            values=dict(width=8,heads=2,graph_layers=1,dilations=(1,),dropout=0,warm_seconds=1);values.update(kwargs)
            return Config(**values)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fp,ap,gp,tp,vp=(root/n for n in ('base.npz','audio.npz','graph.npz','train.csv','validation.csv'))
            base.save(fp);audio.save(ap);graph.save(gp);write_labels(tp,train);write_labels(vp,validation)
            audit=root/'audit.json';audit.write_text(json.dumps(dict(train_labels_sha256=sha(tp),validation_labels_sha256=sha(vp))))
            source=root/'original'
            with patch('relational_possession.experiment.Config',side_effect=small):
                original_compare(Namespace(features=str(fp),audio=str(ap),graph=str(gp),train_labels=str(tp),validation_labels=str(vp),
                    audit=str(audit),out=str(source),epochs=1,seeds=[0,1,2],device='cpu',bootstrap=0,compare=[],cross_validate=True,calibrate=False))
            trees=root/'trees';trees.mkdir()
            plan=json.loads((source/'experiment.json').read_text())
            (trees/'experiment.json').write_text(json.dumps(dict(fingerprints=dict(features=sha(fp),audio=sha(ap),train_labels=sha(tp),validation_labels=sha(vp)))))
            for relative in ('global','cv/fold_0','cv/fold_1'):
                (trees/relative).mkdir(parents=True)
                with np.load(source/relative/'graph_tcn/scores.npz') as data:
                    np.savez_compressed(trees/relative/'trees_scores.npz',probability=data['probability'],boundary=data['boundary'])
            hashes={str(p.relative_to(source)):sha(p) for p in source.glob('**/*') if p.is_file()}
            out=root/'matched';calls=[]
            def guarded(actual,prediction,bootstrap=0):
                if np.array_equal(actual,truth):self.assertTrue((out/'decision.json').exists());calls.append(1)
                return metrics(actual,prediction,bootstrap)
            with patch('relational_possession.matched.Config',side_effect=small),patch('relational_possession.matched.metrics',side_effect=guarded):
                compare(Namespace(features=str(fp),audio=str(ap),graph=str(gp),source_study=str(source),boosted_study=str(trees),
                    train_labels=str(tp),validation_labels=str(vp),out=str(out),epochs=1,device='cpu',bootstrap=0))
            self.assertEqual(len(calls),1)
            for name,digest in hashes.items():self.assertEqual(sha(source/name),digest)
            report=json.loads((out/'report.json').read_text());self.assertEqual(len(report['decision']['oof_scores']),8)
            for i in (0,1):
                folder=out/'cv'/f'fold_{i}'/'graph_tcn'
                self.assertEqual((folder/'seed_0.pt').read_bytes(),(source/'cv'/f'fold_{i}'/'graph_tcn/seed_0.pt').read_bytes())
                info=json.loads((folder/'model.json').read_text());self.assertEqual(info['seeds'],[0,1,2])
            result=root/'regenerated.csv'
            moved_graph,moved_trees=root/'relocated_graph',root/'relocated_trees'
            shutil.copytree(source,moved_graph);shutil.copytree(trees,moved_trees)
            with patch('relational_possession.matched.read_labels',side_effect=AssertionError('Reference read')):
                predict(Namespace(features=str(fp),audio=str(ap),graph=str(gp),source_study=str(moved_graph),boosted_study=str(moved_trees),
                    model=str(out),out=str(result),device='cpu'))
            self.assertEqual(result.read_bytes(),(out/'selected_predictions.csv').read_bytes())


if __name__=='__main__':unittest.main()
