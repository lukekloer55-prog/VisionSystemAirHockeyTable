"""Train a CPU PPO policy in the simulation; no real motor commands are sent."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shutil

from airhockey_env import AirHockeyEnv, Settings, TASK_SCHEMA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=300000,help='Additional environment steps to train')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--config',type=Path,help='JSON file with measured physical parameters')
    parser.add_argument('--reward-config',type=Path,help='Override reward weights for a new or resumed run')
    parser.add_argument('--run-dir',type=Path,help='New output folder; default is a timestamped runs/ folder')
    parser.add_argument('--resume',type=Path,help='Checkpoint ZIP to continue; uses its adjacent settings.json')
    parser.add_argument('--threads',type=int,default=2,help='CPU threads for the small policy network')
    parser.add_argument('--reward-ui',action='store_true',help='Open local browser sliders for live reward editing')
    parser.add_argument('--render',action='store_true',help='Watch the actual training rollouts in a window')
    parser.add_argument('--render-speed',type=float,default=1.0,
                        help='Playback speed while training with --render: 1 is real time, 4 is four times faster')
    args=parser.parse_args()
    if args.steps<=0 or args.threads<=0:
        parser.error('steps and threads must be positive')
    if not math.isfinite(args.render_speed) or args.render_speed<=0:
        parser.error('render-speed must be positive and finite')
    # Imports here keep the basic movement demo independent of PyTorch.
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from training_callbacks import SnapshotCheckpoint, BestSnapshot, RewardPanelCallback, snapshot
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    torch.set_num_threads(args.threads)
    if args.resume and args.config:
        parser.error('Resume uses the original settings; do not also specify --config.')
    if args.resume:
        candidates=[args.resume.with_suffix('.settings.json'),args.resume.parent/'settings.json',args.resume.parent.parent/'settings.json']
        config_path=next((p for p in candidates if p.exists()),None)
        if config_path is None:
            parser.error('Cannot find settings.json beside the checkpoint or its parent folder.')
        schema_path=config_path.with_name('task_schema.json')
        if not schema_path.exists() or json.loads(schema_path.read_text())!=TASK_SCHEMA:
            parser.error('This checkpoint is from a different task. Start a NEW full-rally run without --resume.')
        settings=Settings.load(config_path)
        saved_geometry=config_path.with_name('plotclock_geometry.json')
        current_geometry=Path(__file__).with_name('plotclock_geometry.json')
        if not saved_geometry.exists() or json.loads(saved_geometry.read_text())!=json.loads(current_geometry.read_text()):
            parser.error('Cannot resume: arm geometry differs from the original training run.')
    else:
        settings=Settings.load(args.config)
    if args.reward_config:
        settings=settings.with_rewards(args.reward_config)
    run=args.run_dir or Path('runs')/datetime.now().strftime('rally-%Y%m%d-%H%M%S-%f')
    # Never overwrite an existing training run or checkpoint directory.
    run.mkdir(parents=True,exist_ok=False)
    settings.save(run/'settings.json')
    (run/'task_schema.json').write_text(json.dumps(TASK_SCHEMA,indent=2))
    # Save the geometry used by this run to detect accidental incompatible playback.
    geometry=Path(__file__).with_name('plotclock_geometry.json')
    shutil.copyfile(geometry,run/'plotclock_geometry.json')
    (run/'run.json').write_text(json.dumps({'seed':args.seed,'requested_steps':args.steps,
        'algorithm':'PPO','device':'cpu','resume':str(args.resume) if args.resume else None},indent=2))
    env=Monitor(AirHockeyEnv(settings,render_mode='human' if args.render else None),str(run/'train.monitor.csv'))
    eval_env=Monitor(AirHockeyEnv(settings),str(run/'eval.monitor.csv'))
    eval_env.reset(seed=args.seed+10000)
    model=None;panel=None
    try:
        if args.resume:
            model=PPO.load(args.resume,env=env,device='cpu')
        else:
            model=PPO('MlpPolicy',env,device='cpu',seed=args.seed,verbose=1,
                n_steps=1024,batch_size=64,n_epochs=5,learning_rate=3e-4,
                gamma=settings.gamma,policy_kwargs={'net_arch':[64,64]})
        model.set_logger(configure(str(run),['stdout','csv']))
        callbacks=[SnapshotCheckpoint(env.unwrapped,run/'checkpoints'),
            EvalCallback(eval_env,best_model_save_path=str(run/'best'),log_path=str(run/'evaluation'),
                         eval_freq=10000,n_eval_episodes=10,deterministic=True,
                         callback_on_new_best=BestSnapshot(env.unwrapped,run/'best'))]
        if args.reward_ui:
            from reward_ui import RewardPanel
            panel=RewardPanel(settings,run)
            callbacks.insert(0,RewardPanelCallback(panel,env.unwrapped,eval_env.unwrapped,run,callbacks[1]))
        from rally_metrics import RallyMetricsCallback
        callbacks.append(RallyMetricsCallback())
        if args.render:
            from live_training import LiveTrainingCallback
            callbacks.insert(0,LiveTrainingCallback(env.unwrapped,args.render_speed))
        # Best/checkpoint folders carry settings too, so playback remains self-contained.
        for directory in ('best','checkpoints'):
            (run/directory).mkdir(exist_ok=True)
            settings.save(run/directory/'settings.json')
            (run/directory/'task_schema.json').write_text(json.dumps(TASK_SCHEMA,indent=2))
            shutil.copyfile(geometry,run/directory/'plotclock_geometry.json')
        print(f'Training on CPU. Results: {run.resolve()}')
        print('Ctrl+C stops training and saves latest.zip.')
        if args.render:
            print('Live training window enabled. Closing it also stops and saves.')
            print('The window may pause during neural-network updates and separate evaluation episodes.')
        else:
            print('Headless training. Add --render to watch training in a window.')
        try:
            model.learn(total_timesteps=args.steps,callback=callbacks,reset_num_timesteps=not bool(args.resume))
        except KeyboardInterrupt:
            print('Interrupted; saving current weights.')
        model.save(run/'latest.zip')
        snapshot(run,env.unwrapped.settings,'latest.zip')
        snapshot(run,env.unwrapped.settings)
        print(f'Saved: {run / "latest.zip"}')
        print(f'Watch: uv run --extra train python play_policy.py --model "{run / "latest.zip"}"')
    finally:
        if panel is not None:
            panel.close()
        env.close();eval_env.close()


if __name__=='__main__':
    main()
