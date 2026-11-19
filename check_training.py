"""Run physics and API checks before starting a longer training job."""
import numpy as np
import mujoco
from stable_baselines3.common.env_checker import check_env
from airhockey_env import AirHockeyEnv, Settings
from plotclock_demo import TORQUE, BASE_Y, motor_targets
from table_demo import WIDTH, LENGTH, RADIUS


def place_puck(env,position,velocity):
    # Controlled test setup only: production episodes use randomized reset().
    for i,axis in enumerate(('x','y')):
        env.data.joint('puck_'+axis).qpos[0]=position[i]
        env.data.joint('puck_'+axis).qvel[0]=velocity[i]
    mujoco.mj_forward(env.model,env.data)
    env.previous_potential=env._potential()


def main():
    env=AirHockeyEnv()
    try:
        check_env(env,warn=True)
        a,_=env.reset(seed=7);b,_=env.reset(seed=7)
        np.testing.assert_array_equal(a,b)
        # Ensure initial shots cross a reachable defense strip under the action limits.
        for seed in range(50):
            env.reset(seed=seed)
            p,v=env.puck_state()
            x=p[0]+v[0]*(env.defense_y-p[1])/v[1]
            targets=motor_targets([x,env.defense_y-BASE_Y])
            assert np.max(np.abs(targets)) <= env.settings.action_angle_rad, 'Shot beyond initial action workspace'
        rng=np.random.default_rng(99)
        env.reset(seed=10)
        for _ in range(600):
            obs,reward,t,tr,info=env.step(rng.uniform(-1,1,2))
            assert env.observation_space.contains(obs) and np.isfinite(reward)
            assert info['closure_error_m']<.002
            assert np.max(np.abs(info['torque_nm']))<=TORQUE+1e-8
            assert info['outcome']!='invalid_physics'
            if t or tr:
                env.reset(seed=int(rng.integers(100000)))
        # A puck beyond the near goal must cause a failed terminal episode.
        env.reset(seed=1)
        place_puck(env,[0,-LENGTH/2-RADIUS-.01],[0,-.5])
        _,_,t,tr,info=env.step(env.command/env.settings.action_angle_rad)
        assert t and not tr and info['outcome']=='goal_conceded'
        # Side rail contact must reverse the puck's lateral velocity.
        env.reset(seed=1)
        place_puck(env,[WIDTH/2-RADIUS-.01,.4],[.5,0])
        for _ in range(15):
            env.step(env.command/env.settings.action_angle_rad)
        assert env.puck_state()[1][0]<0,'Puck did not rebound from side rail'
        # A real paddle impact must be observed and send this incoming puck back.
        env.reset(seed=1)
        pad,_=env.paddle_state()
        place_puck(env,pad+[0,env.settings.paddle_diameter_m/2+RADIUS+.02],[0,-.7])
        contact=False;returned=False
        for _ in range(300):
            _,_,t,tr,info=env.step(env.command/env.settings.action_angle_rad)
            contact=contact or info['paddle_contact']
            if t or tr:
                returned=info['outcome']=='returned';break
        assert contact and returned,f'Contact/return failed: {info}'
        # Time limits are truncations (for correct PPO bootstrapping), not goals.
        short=AirHockeyEnv(Settings(episode_seconds=.04))
        try:
            short.reset(seed=1)
            short.step(short.command/short.settings.action_angle_rad)
            _,_,t,tr,info=short.step(short.command/short.settings.action_angle_rad)
            assert tr and not t and info['outcome']=='timeout'
        finally:
            short.close()
        print('PASS: Gymnasium API, seeded reset, reachable shots, random actions, torque/closure, goal, rail rebound, paddle return, timeout.')
    finally:
        env.close()


if __name__=='__main__':
    main()
