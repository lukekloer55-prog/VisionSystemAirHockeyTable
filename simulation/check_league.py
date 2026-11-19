"""League promotion, scheduling, settings and stopped-run validation."""
import unittest
from train_league import promotion,schedule,retain_champion
from skills_controls import controls

class LeagueTests(unittest.TestCase):
 def test_balanced_schedule(self):
  pairs=schedule(['a','b'],['h0','h1'])
  self.assertEqual(len(pairs),9)
  self.assertEqual(len(set(pairs)),len(pairs))
  for player in ['champion','a','b']:self.assertEqual(sum(player in pair for pair in pairs),4)
 def results(self):
  return [{'pair':list(pair),'scores':[0,0]} for pair in schedule(['a','b'],['h'])]
 def set_score(self,results,pair,scores):
  next(r for r in results if r['pair']==pair)['scores']=scores
 def test_ties_keep_champion(self):
  self.assertEqual(promotion(['a','b'],['h'],self.results(),2)['retained'],'champion')
 def test_promotion_needs_head_to_head_and_historical_nonregression(self):
  r=self.results();self.set_score(r,['champion','a'],[0,2]);self.set_score(r,['a','b'],[2,0])
  self.assertEqual(promotion(['a','b'],['h'],r,2)['retained'],'a')
  self.set_score(r,['champion','h'],[2,0]);self.set_score(r,['a','h'],[0,2])
  self.assertEqual(promotion(['a','b'],['h'],r,2)['retained'],'champion')
  self.set_score(r,['champion','h'],[0,0]);self.set_score(r,['a','h'],[2,0]);self.set_score(r,['champion','a'],[0,0])
  self.assertEqual(promotion(['a','b'],['h'],r,2)['retained'],'champion')
 def test_archive_keeps_anchor_and_recent_champions(self):
  state={'champion_id':0,'round':0,'history':['history_000000']};models={'champion':'initial','history_000000':'initial'}
  for index in range(8):
   models['candidate_0']=f'winner-{index}';models['candidate_1']='loser'
   models=retain_champion(state,models,'candidate_0');state['round']+=1
   self.assertEqual(models['history_000000'],'initial');self.assertLessEqual(len(state['history']),5)
   self.assertNotIn('candidate_1',models)
  self.assertEqual(models['champion'],'winner-7')
  self.assertEqual(state['history'],['history_000000','history_000004','history_000005','history_000006','history_000007'])
 def test_league_rewards_match_game(self):self.assertEqual(controls('league'),controls('selfplay'))
if __name__=='__main__':unittest.main()
