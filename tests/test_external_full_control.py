from pathlib import Path
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from external_possession.full_control import control_labels,load_fifa,read_events


class FullControlTests(unittest.TestCase):
    def test_loss_after_recorded_recovery_does_not_erase_winning_control(self):
        periods=np.r_[0,np.ones(59),np.full(60,2)].astype(np.int8)
        rows=[['Away','SET PIECE','KICK OFF',1,5,0],['Away','BALL LOST','INTERCEPTION',1,15,20],
              ['Home','RECOVERY','INTERCEPTION',1,18,18],['Home','FAULT RECEIVED','',1,30,30],
              ['Home','SET PIECE','FREE KICK',1,40,40],['Home','PASS','',1,45,50],
              ['Home','BALL OUT','',1,55,58],['Away','SET PIECE','KICK OFF',2,70,70]]
        events=pd.DataFrame(rows,columns=['Team','Type','Subtype','Period','Start Frame','End Frame'])
        codes,stats=control_labels(events,periods)
        self.assertTrue(np.all(codes[5:18]==1) and np.all(codes[18:30]==0))
        self.assertTrue(np.all(codes[30:40]==2) and np.all(codes[40:58]==0))
        self.assertTrue(np.all(codes[58:70]==2));self.assertEqual(codes[0],3)
        self.assertTrue(np.all(codes[70:]==3))  # no resolved live tail is invented

    def test_same_frame_conflict_stays_unknown_and_stop_has_priority(self):
        periods=np.r_[0,np.ones(49)].astype(np.int8)
        rows=[['Home','RECOVERY','',1,10,10],['Away','PASS','',1,10,20],
              ['Home','SET PIECE','',1,25,25],['Home','PASS','',1,25,30],
              ['Away','BALL OUT','',1,25,25],['Home','PASS','OFFSIDE',1,35,40]]
        events=pd.DataFrame(rows,columns=['Team','Type','Subtype','Period','Start Frame','End Frame'])
        codes,stats=control_labels(events,periods)
        self.assertEqual(stats['conflicting_control_instants'],1);self.assertTrue(np.all(codes[10:20]==3))
        self.assertEqual(codes[25],3)  # contradictory closed pass is censored, not relabelled
        self.assertTrue(np.all(codes[40:]==2))

    def test_json_schema_preserves_frame_ids_and_subtype_lists(self):
        event=dict(team=dict(id='FIFATMA'),type=dict(name='SHOT'),subtypes=[dict(name='ON TARGET'),dict(name='GOAL')],
                   period=2,start=dict(frame=100),end=dict(frame=120))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'events.json';path.write_text(json.dumps(dict(data=[event])))
            row=read_events(path).iloc[0]
        self.assertEqual(row['Team'],'Home');self.assertEqual(row['Start Frame'],100);self.assertEqual(row['End Frame'],120)
        self.assertEqual(row['Subtype'],'ON TARGET-GOAL')

    def test_fifa_roster_sections_map_substitutions_and_keep_frame_clock(self):
        root=ET.Element('EPTS');meta=ET.SubElement(root,'Metadata');global_=ET.SubElement(meta,'GlobalConfig')
        ET.SubElement(global_,'FrameRate').text='25';params=ET.SubElement(global_,'ProviderGlobalParameters')
        for name,value in [('first_half_start',1),('first_half_end',2),('second_half_start',3),('second_half_end',4)]:
            p=ET.SubElement(params,'ProviderParameter');ET.SubElement(p,'Name').text=name;ET.SubElement(p,'Value').text=str(value)
        players=ET.SubElement(meta,'Players');channels=ET.SubElement(meta,'PlayerChannels')
        for pid,team,gk in [('h0','FIFATMA',True),('h1','FIFATMA',False),('h2','FIFATMA',False),('a0','FIFATMB',True),('a1','FIFATMB',False)]:
            p=ET.SubElement(players,'Player',id=pid,teamId=team);ET.SubElement(p,'Name').text=pid
            groups=ET.SubElement(p,'ProviderPlayerParameters');v=ET.SubElement(groups,'ProviderParameter')
            ET.SubElement(v,'Name').text='position_type';ET.SubElement(v,'Value').text='Goalkeeper' if gk else 'Midfielder'
            for axis in ('x','y'):ET.SubElement(channels,'PlayerChannel',id=pid+'_'+axis,playerId=pid,channelId=axis)
        formats=ET.SubElement(root,'DataFormatSpecifications')
        for first,last,home in [(1,2,'h1'),(3,4,'h2')]:
            spec=ET.SubElement(formats,'DataFormatSpecification',startFrame=str(first),endFrame=str(last),separator=':')
            group=ET.SubElement(spec,'SplitRegister',separator=';')
            for pid in ('h0',home,'a0','a1'):
                pair=ET.SubElement(group,'SplitRegister',separator=',')
                for axis in ('x','y'):ET.SubElement(pair,'PlayerChannelRef',playerChannelId=pid+'_'+axis)
        with tempfile.TemporaryDirectory() as tmp:
            xml,txt=Path(tmp)/'meta.xml',Path(tmp)/'track.txt';ET.ElementTree(root).write(xml)
            txt.write_text(''.join(f'{i}:0.1,0.5;0.4,0.5;0.9,0.5;0.6,0.5:0.5,0.5\n' for i in range(1,5)))
            match,periods,keepers=load_fifa(xml,txt)
        self.assertEqual(match.halves,((1,1,2),(2,3,4)));np.testing.assert_array_equal(periods,[0,1,1,2,2])
        self.assertTrue(np.isnan(match.players[3,0]).all());self.assertAlmostEqual(float(match.players[3,1,0]),42,places=4)
        self.assertAlmostEqual(float(match.players[1,23,0]),10.5,places=4);self.assertAlmostEqual(float(match.players[1,45,0]),94.5,places=4)
        self.assertEqual(match.team_by_frame()[3,1],0);self.assertEqual(match.team_by_frame()[3,24],1)


if __name__=='__main__':unittest.main()
