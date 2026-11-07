from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from learned_possession.features import FeatureCache
from possession.phase_hint import phase_hints,apply,POLICIES


class PhaseHintTests(unittest.TestCase):
    def test_territorial_tags_conflicts_and_half_crossings_cannot_create_hints(self):
        root=ET.Element('file');events=ET.SubElement(root,'ALL_INSTANCES')
        def add(code,team,a,b,half):
            event=ET.SubElement(events,'instance')
            for key,value in [('code',code),('start',str(a)),('end',str(b))]:ET.SubElement(event,key).text=value
            for group,text in [('Team',team),('Half',half)]:
                label=ET.SubElement(event,'label');ET.SubElement(label,'group').text=group;ET.SubElement(label,'text').text=text
        add('BUILD UP','Barcelona',1,8,'1st Half');add('PROGRESSION','Barcelona',2,6,'1st Half')
        add('LONG BALL','Atletic Club',4,7,'1st Half');add('BALL IN FINAL THIRD','Atletic Club',1,4,'1st Half')
        add('DEFENDING IN MIDDLE THIRD','Atletic Club',1,4,'1st Half');add('BUILD UP','Barcelona',8,24,'1st Half')
        add('CREATING CHANCES','Atletic Club',22,26,'2nd Half');add('BUILD UP','N/A',0,30,'1st Half')
        base=FeatureCache(np.zeros((30,1)),('motion',),1,((1,0,9),(2,20,29)),{})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'tags.xml';ET.ElementTree(root).write(path);hint,stats=phase_hints(path,base)
        np.testing.assert_array_equal(hint[:10],[-1,0,0,0,-1,-1,-1,0,0,0])
        np.testing.assert_array_equal(hint[10:22],np.full(12,-1));np.testing.assert_array_equal(hint[22:26],np.ones(4))
        self.assertEqual(stats['conflict_frames'],3)

    def test_hint_changes_only_missing_observation_owner_and_preserves_dead_mass(self):
        p=np.array([[.8,.1,.1],[.2,.5,.3],[0,.1,.9],[.4,.3,.3]],np.float32)
        hint=np.array([1,0,0,-1]);missing=np.array([True,True,False,True])
        only=apply('hint_only',p,hint,missing)
        np.testing.assert_allclose(only[0],[.225,.675,.1],atol=1e-7)
        np.testing.assert_array_equal(only[:,2],p[:,2]);np.testing.assert_array_equal(only[2:],p[2:])
        for policy in POLICIES:
            result=apply(policy,p,hint,missing);swapped_hint=np.where(hint<0,-1,1-hint)
            swapped=apply(policy,p[:,[1,0,2]],swapped_hint,missing)
            np.testing.assert_allclose(result[:,[1,0,2]],swapped,atol=1e-7)
            np.testing.assert_allclose(result.sum(1),1,atol=1e-7)
        np.testing.assert_array_equal(apply('control',p,hint,missing),p)


if __name__=='__main__':unittest.main()
