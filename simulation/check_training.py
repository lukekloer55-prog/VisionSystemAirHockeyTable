"""Physics, reward-event and Gymnasium regression tests for full rallies."""
import numpy as np
import mujoco
from stable_baselines3.common.env_checker import check_env
from airhockey_env import AirHockeyEnv, Settings
from plotclock_demo import TORQUE, BASE_Y, motor_targets
from table_demo import WIDTH, LENGTH, RADIUS


def place_puck(env,position,velocity):
    """Controlled test setup; normal episodes use randomized reset()."""
    for i,axis in enumerate(('x','y')):
        env.data.joint('puck_'+axis).qpos[0]=position[i]
        env.data.joint('puck_'+axis).qvel[0]=velocity[i]
    mujoco.mj_forward(env.model,env.data)
    env.previous_potential=env._potential()


def hold(env):
    return env.step(env.command/env.settings.action_angle_rad)


def main():
    env=AirHockeyEnv()
    try:
        check_env(env,warn=True)
        a,_=env.reset(seed=7);b,_=env.reset(seed=7)
        np.testing.assert_array_equal(a,b)
        assert a.shape==(25,)
        for seed in range(50):
            env.reset(seed=seed)
            p,v=env.puck_state()
            x=p[0]+v[0]*(env.defense_y-p[1])/v[1]
            targets=motor_targets([x,env.defense_y-BASE_Y])
            assert np.max(np.abs(targets))<=env.settings.action_angle_rad
        rng=np.random.default_rng(99);env.reset(seed=10)
        for _ in range(600):
            obs,reward,t,tr,info=env.step(rng.uniform(-1,1,2))
            assert env.observation_space.contains(obs) and np.isfinite(reward)
            assert np.isclose(reward,sum(info['reward_components'].values()))
            assert info['closure_error_m']<.002
            assert np.max(np.abs(info['torque_nm']))<=TORQUE+1e-8
            assert info['outcome']!='invalid_physics'
            if t or tr:
                env.reset(seed=int(rng.integers(100000)))
        # Both actual goal openings produce terminal outcomes and the right rewards.
        for sign,outcome,term in [(1,'goal_scored','goal'),(-1,'goal_conceded','conceded_goal')]:
            env.reset(seed=1)
            place_puck(env,[0,sign*(LENGTH/2+RADIUS+.01)],[0,sign*.5])
            _,_,t,tr,info=hold(env)
            assert t and not tr and info['outcome']==outcome
            assert info['reward_components'][term]==sign*30
        # Crossing an end plane outside the goal is an escape, never a scored goal.
        env.reset(seed=1)
        place_puck(env,[.3,LENGTH/2+RADIUS+.05],[0,.5])
        _,_,t,_,info=hold(env)
        assert t and info['outcome']=='puck_escaped' and info['reward_components']['goal']==0
        # A side-wall rebound is physical but earns no bank bonus by itself.
        env.reset(seed=1);env.touched=True
        place_puck(env,[WIDTH/2-RADIUS-.01,.4],[.5,0])
        for _ in range(15):
            _,_,t,tr,info=hold(env)
            assert info['reward_components']['bank_goal']==0
        assert env.puck_state()[1][0]<0 and env.bank_since_contact
        place_puck(env,[0,LENGTH/2+RADIUS+.01],[0,.5])
        _,_,t,_,info=hold(env)
        assert t and info['reward_components']['bank_goal']==3
        # A real impact earns a useful-hit/save reward, then crossing does not stop play.
        env.reset(seed=1);pad,_=env.paddle_state()
        place_puck(env,pad+[0,env.settings.paddle_diameter_m/2+RADIUS+.02],[0,-.7])
        hit=False;crossed=False
        for _ in range(500):
            _,_,t,tr,info=hold(env)
            hit=hit or info['useful_hit']
            if info['center_line_crossing']:
                assert not t and not tr and info['reward_components']['center_line']>0
                crossed=True;break
            assert not (t or tr),info
        assert hit and crossed
        # A puck staying in the far half does not repeatedly pay center-line rewards.
        for _ in range(5):
            _,_,_,_,info=hold(env)
            assert info['reward_components']['center_line']==0
        # Another inbound shot in the SAME rally can be contacted and rewarded again.
        pad,_=env.paddle_state()
        place_puck(env,pad+[0,env.settings.paddle_diameter_m/2+RADIUS+.02],[0,-.7])
        second_hit=False
        for _ in range(40):
            _,_,t,tr,info=hold(env)
            if info['useful_hit']:
                assert info['reward_components']['contact']>0
                second_hit=True;break
            assert not(t or tr)
        assert second_hit and env.hit_count>=2
        # Repeated event payments cannot exceed the rally budget.
        remaining=env.bonus_remaining
        assert np.isclose(sum(env._event_bonus(2) for _ in range(20)),remaining)
        assert env.bonus_remaining==0
        # Crossing without a paddle impact does not qualify for a center-line reward.
        env.reset(seed=1);place_puck(env,[0,RADIUS-.005],[0,.5])
        _,_,t,tr,info=hold(env)
        assert not t and not tr and info['reward_components']['center_line']==0
        # Behind penalty is time-scaled, and pursuit remains active after contact.
        env.reset(seed=1);pad,_=env.paddle_state();env.touched=True
        place_puck(env,pad+[0,-.20],[0,0])
        _,_,_,_,info=hold(env)
        assert info['behind_paddle'] and np.isclose(info['reward_components']['behind_paddle'],-.04)
        assert np.all(np.isfinite(info['chase_target']))
        # A target in the reachable own half follows the puck rather than a fixed line.
        first=env.chase_target().copy()
        place_puck(env,[.2,-.5],[0,0])
        assert np.linalg.norm(first-env.chase_target())>.01
        # Timeout is a truncation rather than a goal.
        short=AirHockeyEnv(Settings(episode_seconds=.04))
        try:
            short.reset(seed=1);hold(short)
            _,_,t,tr,info=hold(short)
            assert tr and not t and info['outcome']=='timeout'
            assert info['reward_components']['timeout']==-5
        finally:
            short.close()
        print('PASS: API/reset, reachable starts, random actions, torque/closure, both goals, escapes, bank goal, continuous rally, repeat hits, bonus cap, center gating, chase/behind rewards and timeout.')
    finally:
        env.close()


if __name__=='__main__':
    main()
