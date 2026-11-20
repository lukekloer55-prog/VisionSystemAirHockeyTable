"""Render sampled live CPU/GPU states or replay a saved shared movement policy."""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco
from skills_env import SkillsEnv,Curriculum
from airhockey_env import Settings
from hybrid_movement import apply_proxy

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--live',action='store_true');args=p.parse_args()
    run=args.run_dir;meta=json.loads((run/'mjx_run.json').read_text());envs=[]
    def create():
        e=SkillsEnv('movement',Settings(**meta.get('settings',{'action_angle_rad':.30})),Curriculum(**meta.get('curriculum',{})),'human');apply_proxy(e);e.reset();envs.append(e);return e
    try:
        if args.live:
            (run/'WATCH').touch();last=0
            print('Live sampled worlds: GPU 1, GPU 2, then CPU 1/2 if enabled. Rendering does not advance or train these copies.',flush=True)
            while True:
                if any(e.viewer is not None and not e.viewer.is_running() for e in envs):break
                try:
                    with np.load(run/'live.npz',allow_pickle=False) as f:frames={k:f[k].copy() for k in f.files}
                except (FileNotFoundError,ValueError,OSError):time.sleep(.2);continue
                if float(frames['updated'])!=last:
                    last=float(frames['updated'])
                    while len(envs)<len(frames['qpos']):create()
                    for i,e in enumerate(envs):
                        if i>=len(frames['qpos']):break
                        e.data.qpos[:]=frames['qpos'][i];e.data.qvel[:]=frames['qvel'][i];e.data.mocap_pos[0]=[*frames['target'][i],.025];mujoco.mj_forward(e.model,e.data)
                for i,e in enumerate(envs):
                    e.render()
                    age=time.time()-float(frames['updated'])
                    e.viewer.set_texts((None,None,str(frames['labels'][i]),'Live sampled state' if age<2 else 'Waiting for next state / trainer stopped'))
                time.sleep(.03)
        else:
            from stable_baselines3 import PPO
            e=create();episode=0
            while True:
                model=PPO.load(run/'latest.zip',device='cpu');obs,_=e.reset(seed=episode);episode+=1
                while True:
                    began=time.monotonic();obs,_,term,trunc,info=e.step(model.predict(obs,deterministic=True)[0])
                    if e.viewer is not None and not e.viewer.is_running():return
                    e.render();e.viewer.set_texts((None,None,'Saved policy replay','CPU visualization, no learning'));time.sleep(max(0,e.settings.control_dt-(time.monotonic()-began)))
                    if term or trunc:print(info['outcome'],flush=True);break
    except KeyboardInterrupt:pass
    finally:
        for e in envs:e.close()
        if args.live:(run/'WATCH').unlink(missing_ok=True)
if __name__=='__main__':main()
