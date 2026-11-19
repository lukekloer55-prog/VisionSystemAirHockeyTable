"""Verify CPU policy import into shared trainer without requiring CUDA."""
import tempfile,json,sys,hashlib
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from skills_env import SkillsEnv,SCHEMA,Curriculum
from airhockey_env import Settings
from train_skills import Store
import train_mjx_movement
class TestGPU(DummyVecEnv):
    def __init__(self,count,device,seed,settings):super().__init__([lambda:SkillsEnv('movement',settings) for _ in range(count)])
def main():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
        root=Path(tmp);src=root/'cpu';src.mkdir();dest=root/'shared'
        physical=Settings(paddle_mass_kg=.21)
        source_env=DummyVecEnv([lambda:SkillsEnv('movement',physical)])
        source=PPO('MlpPolicy',source_env,n_steps=2,batch_size=2,n_epochs=1,device='cpu',policy_kwargs={'net_arch':[8,8]})
        source.learn(2)
        state={'schema':SCHEMA,'mode':'movement','settings':asdict(physical),'curriculum':asdict(Curriculum()),'geometry':json.loads(Path(__file__).with_name('plotclock_geometry.json').read_text())}
        folder=Store(src).save(state,{'champion':source});before=hashlib.sha256((folder/'champion.zip').read_bytes()).hexdigest()
        with patch('mjx_movement.MJXMovement',TestGPU),patch.object(sys,'argv',['train_mjx_movement.py','--run-dir',str(dest),'--initial-run',str(src),'--num-envs','2','--steps','4']):train_mjx_movement.main()
        meta=json.loads((dest/'mjx_run.json').read_text());assert meta['settings']['paddle_mass_kg']==.21 and meta['total_envs']==2 and meta['lineage']['source_run']==str(src.resolve())
        loaded=PPO.load(dest/'latest.zip',device='cpu');assert loaded.num_timesteps==6
        assert hashlib.sha256((folder/'champion.zip').read_bytes()).hexdigest()==before
        source_env.close()
        print('PASS CPU policy/optimizer import, copied physical settings, continued steps, untouched source, shared checkpoint')
if __name__=='__main__':main()
