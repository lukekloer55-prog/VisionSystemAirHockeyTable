import tempfile,json,io,zipfile,os
from pathlib import Path
from unittest.mock import patch,MagicMock
from dataclasses import asdict
from skills_env import Curriculum
from skills_controls import controls
from training_ui import TrainingController
import gpu_panel
with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
 c=TrainingController(tmp);c.code=Path(tmp)/'code';(c.code/'.venv-mjx'/('Scripts' if os.name=='nt' else 'bin')).mkdir(parents=True);(c.code/'.venv-mjx'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')).touch()
 vals={k:asdict(Curriculum())[k] for k in controls('movement')}
 process=MagicMock();process.poll.return_value=None
 with patch('gpu_panel.subprocess.Popen',return_value=process) as pop,patch('gpu_panel.threading.Thread'):
  gpu_panel.action(c,'start',{'name':'new','num_envs':64,'values':vals})
  args=pop.call_args.args[0];assert '--curriculum' in args and '--num-envs' in args
 c.process=None
 p=Path(tmp)/'old';p.mkdir();(p/'mjx_run.json').write_text(json.dumps({'schema':'mjx-movement-experimental-v1'}));(p/'latest.zip').write_bytes(b'test')
 assert gpu_panel.state(c)['runs'][0]['name']=='old'
 with patch.object(c,'running',return_value=True):
  gpu_panel.action(c,'stop',{'name':'old'});gpu_panel.action(c,'save',{'name':'old'})
 assert (p/'STOP').exists() and (p/'SAVE').exists()
 gpu_panel.action(c,'rewards',{'name':'old','values':vals});assert (p/'pending_rewards.json').exists()
 with zipfile.ZipFile(io.BytesIO(gpu_panel.export(c,'old'))) as z:assert 'old/latest.zip' in z.namelist()
 with patch('gpu_panel.subprocess.Popen',return_value=process) as pop,patch('gpu_panel.threading.Thread'):
  gpu_panel.action(c,'start',{'name':'old','num_envs':256,'resume':True});assert '--resume' in pop.call_args.args[0]
 print('PASS GPU new/resume command, run discovery, stop/save, reward queue and export')
