"""Dashboard API, transaction and checkpoint tests; no graphical display needed."""
from dataclasses import asdict
import io,json,tempfile,unittest,urllib.request,urllib.error,zipfile
from pathlib import Path
from unittest.mock import patch,MagicMock
from skills_env import SkillsEnv,Curriculum,SCHEMA
from skills_controls import controls,validate,atomic_json,apply_pending
from training_ui import TrainingController,Dashboard
from train_skills import Store,RunLock

class Fake:
 def save(self,path):Path(path).write_bytes(b'model')

class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=Path.cwd());self.root=Path(self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def make_run(self,mode='movement'):
  run=self.root/'example';run.mkdir();store=Store(run)
  state=dict(schema=SCHEMA,mode=mode,curriculum=asdict(Curriculum()),phase='train_a',trained=0,total_steps=0,round=0)
  models={'champion':Fake()};store.save(state,models)
  return run,store,state,models
 def test_reward_transaction_and_fair_round(self):
  run,store,state,models=self.make_run();env=SkillsEnv()
  values={k:getattr(env.curriculum,k) for k in controls('movement')};values['time_cost']=1.25;values['min_tip_angle_deg']=70
  atomic_json(run/'pending_rewards.json',{'revision':'one','values':values})
  apply_pending(env,state,models,store)
  self.assertEqual(env.curriculum.time_cost,1.25);self.assertEqual(env.curriculum.min_tip_angle_deg,70)
  self.assertEqual(store.load()[0]['curriculum']['time_cost'],1.25);self.assertFalse((run/'pending_rewards.json').exists())
  state['mode']='selfplay';state['phase']='train_b'
  game={k:getattr(env.curriculum,k) for k in controls('selfplay')};game['goal_reward']=55
  atomic_json(run/'pending_rewards.json',{'revision':'two','values':game})
  self.assertIsNone(apply_pending(env,state,models,store));self.assertNotEqual(env.curriculum.goal_reward,55)
  state['phase']='train_a';apply_pending(env,state,models,store)
  self.assertEqual(env.curriculum.goal_reward,55);env.close()
 def test_validation(self):
  values={k:getattr(Curriculum(),k) for k in controls('movement')}
  for value in [-1,float('nan'),True,1000]:
   with self.assertRaises(ValueError):validate('movement',{**values,'time_cost':value})
  with self.assertRaises(ValueError):validate('movement',{})
 def test_lock_and_save_request(self):
  run,store,state,models=self.make_run();controller=TrainingController(self.root)
  with RunLock(run/'trainer.lock'):
   self.assertTrue(controller.running(run));self.assertTrue(controller.running(run))
  self.assertEqual((run/'trainer.lock').stat().st_size,1)
  self.assertFalse(controller.running(run))
  (run/'SAVE').touch();store.save(state,models);self.assertFalse((run/'SAVE').exists())
 def test_rename_preserves_checkpoint_and_export(self):
  run,store,state,models=self.make_run();controller=TrainingController(self.root)
  before=(run/'CURRENT.json').read_bytes();_,folder=store.load()
  weights=(folder/'champion.zip').read_bytes()
  self.assertEqual(controller.state()['runs'][0]['display_name'],'example')
  with RunLock(run/'trainer.lock'):
   controller.action('rename',{'name':'example','display_name':' Robot v1 - movement + self-play '})
  self.assertEqual(controller.state()['runs'][0]['display_name'],'Robot v1 - movement + self-play')
  self.assertEqual((run/'CURRENT.json').read_bytes(),before)
  self.assertEqual((folder/'champion.zip').read_bytes(),weights)
  with zipfile.ZipFile(io.BytesIO(controller.export('example'))) as z:
   self.assertEqual(json.loads(z.read('example/run_details.json'))['display_name'],'Robot v1 - movement + self-play')
  for label in ['', '   ', 'x'*101, 'bad\nname', None]:
   with self.assertRaises(ValueError):controller.action('rename',{'name':'example','display_name':label})
  with self.assertRaises(ValueError):controller.action('rename',{'name':'../outside','display_name':'Bad'})
 def test_league_launch_and_budget(self):
  controller=TrainingController(self.root)
  values={k:getattr(Curriculum(),k) for k in controls('league')}
  request=dict(name='league',mode='league',num_envs=4,candidates=2,values=values,render=False)
  fake=MagicMock();fake.poll.return_value=None
  with patch('training_ui.os.cpu_count',return_value=8),patch('training_ui.subprocess.Popen',return_value=fake) as launch,patch('training_ui.threading.Thread'):
   with self.assertRaises(ValueError):controller.start({**request,'num_envs':3})
   controller.start(request)
   command=launch.call_args.args[0]
   self.assertTrue(any(arg.endswith('train_league.py') for arg in command))
   self.assertEqual(command[command.index('--num-envs')+1],'4')
   self.assertEqual(command[command.index('--candidates')+1],'2')
   self.assertNotIn('--render',command)
 def test_league_rewards_wait_for_round(self):
  run,store,state,models=self.make_run('league');env=SkillsEnv('game')
  try:
   state['phase']='league_train'
   values={k:getattr(Curriculum(),k) for k in controls('league')};values['goal_reward']=55
   atomic_json(run/'pending_rewards.json',{'revision':'league','values':values})
   self.assertIsNone(apply_pending(env,state,models,store))
   state['phase']='league_start';apply_pending(env,state,models,store)
   self.assertEqual(env.curriculum.goal_reward,55)
  finally:env.close()
 def test_http_and_export(self):
  run,store,state,models=self.make_run();controller=TrainingController(self.root);dashboard=Dashboard(controller)
  try:
   def request(route,data=None,token=True,origin=None):
    headers={'Content-Type':'application/json'}
    if token:headers['X-Panel-Token']=dashboard.token
    if origin:headers['Origin']=origin
    req=urllib.request.Request(dashboard.url+route,data=json.dumps(data).encode() if data is not None else None,headers=headers)
    return urllib.request.urlopen(req)
   with request('/') as response:self.assertIn(b'Train, tune, and continue',response.read())
   with request('/api/state') as response:self.assertEqual(json.load(response)['runs'][0]['mode'],'movement')
   values={k:getattr(Curriculum(),k) for k in controls('movement')};values['reach_reward']=22
   with self.assertRaises(urllib.error.HTTPError) as error:request('/api/rewards',{'name':'example','values':values},False)
   self.assertEqual(error.exception.code,403)
   with self.assertRaises(urllib.error.HTTPError):request('/api/rewards',{'name':'example','values':values},origin='https://elsewhere.invalid')
   with request('/api/rewards',{'name':'example','values':values}) as response:self.assertEqual(response.status,200)
   self.assertEqual(json.loads((run/'pending_rewards.json').read_text())['values']['reach_reward'],22)
   with self.assertRaises(urllib.error.HTTPError):request('/api/save',{'name':'../outside'})
   with request('/api/export?name=example') as response:
    with zipfile.ZipFile(io.BytesIO(response.read())) as z:
     self.assertIn('example/CURRENT.json',z.namelist());self.assertIsNone(z.testzip())
  finally:dashboard.close()

if __name__=='__main__':unittest.main()
