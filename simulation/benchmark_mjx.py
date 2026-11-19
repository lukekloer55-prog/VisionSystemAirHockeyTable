"""Optional MJX-JAX physics benchmark. Does not change or train saved policies."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

# Avoid reserving most VRAM before the benchmark decides its batch size.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import mujoco
import numpy as np
from skills_env import SkillsEnv
from plotclock_demo import ARM_Z,TORQUE
from table_demo import WIDTH,RADIUS


def prepare_scene(proxy=False,scenario='rail'):
    env=SkillsEnv('game');env.reset(seed=7)
    model,data=env.model,env.data
    changes=[]
    if proxy:
        model.geom_condim[:]=1
        changes.append('Contact dimensions set to 1: explicitly frictionless normal contact, matching zero friction coefficients')
        # Modify only this temporary compiled model. Preserve compiled body masses/inertias.
        for name in ('puck_shape','paddle','opp_paddle'):
            geom=model.geom(name);radius=float(geom.size[0])
            geom.type=mujoco.mjtGeom.mjGEOM_SPHERE
            geom.size[:]=[radius,0,0]
            if name!='puck_shape':geom.pos[2]=.006-ARM_Z
            model.geom_rbound[geom.id]=radius
            model.geom_aabb[geom.id]=[0,0,0,radius,radius,radius]
            changes.append(name+': cylinder replaced by same-radius sphere; collision centers aligned at z=6mm')
    mujoco.mj_forward(model,data)
    if scenario=='rail':
        data.joint('puck_x').qpos[0]=WIDTH/2-RADIUS-.01
        data.joint('puck_y').qpos[0]=-.3
        data.joint('puck_x').qvel[0]=.8;data.joint('puck_y').qvel[0]=.1
    elif scenario=='paddle':
        pad=data.site('left_tip').xpos
        data.joint('puck_x').qpos[0]=pad[0]+env.settings.paddle_diameter_m/2+RADIUS+.01
        data.joint('puck_y').qpos[0]=pad[1]
        data.joint('puck_x').qvel[0]=-.8;data.joint('puck_y').qvel[0]=0
    mujoco.mj_forward(model,data)
    joints=model.actuator_trnid[:,0]
    qids=model.jnt_qposadr[joints];vids=model.jnt_dofadr[joints]
    targets=data.qpos[qids].copy()
    return env,qids,vids,targets,changes


def cpu_rollout(model,initial,qids,vids,targets,steps):
    import copy
    data=copy.copy(initial)
    began=time.perf_counter();peak_gap=0.;contacts=0
    for _ in range(steps):
        data.ctrl[:]=np.clip(25*(targets-data.qpos[qids])-4*data.qvel[vids],-TORQUE,TORQUE)
        mujoco.mj_step(model,data)
        contacts+=int(data.ncon>0)
        for prefix in ('','opp_'):
            peak_gap=max(peak_gap,float(np.linalg.norm(data.site(prefix+'left_tip').xpos-data.site(prefix+'right_tip').xpos)))
    seconds=time.perf_counter()-began
    return data,{'physics_steps_per_second':steps/seconds,'peak_closure_gap_m':peak_gap,'steps_with_contact':contacts,
                 'finite':bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['gpu','cpu'],default='gpu')
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[16])
    p.add_argument('--steps',type=int,default=200,help='Physics steps per world per timed repeat (1ms each)')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--planar-proxy',action='store_true',help='Explicitly replace puck/paddle collision cylinders with coplanar spheres')
    p.add_argument('--scenario',choices=['rail','paddle'],default='rail')
    p.add_argument('--output',type=Path,default=Path('mjx-benchmark.json'))
    args=p.parse_args()
    if min(args.batch_sizes)<1 or max(args.batch_sizes)>1024 or args.steps<1 or args.repeats<1:p.error('Positive counts required; at most 1024 worlds')
    report={'benchmark':'physics-only, fixed servo targets; no PPO/rewards/rendering','scenario':args.scenario,'proxy':args.planar_proxy,'mujoco':mujoco.__version__,'batches':[]}
    env=None
    try:
        import jax
        import jax.numpy as jnp
        from mujoco import mjx
        report['jax']=jax.__version__;devices=jax.devices(args.device)
        if not devices:raise RuntimeError('Requested device unavailable; CPU fallback is not allowed')
        device=devices[0];report['device']=str(device);report['device_kind']=device.device_kind
        print('Device:',device,'/',device.device_kind,flush=True)
        with jax.default_device(device):
            # Prove kernels execute; enumerating a GPU is not sufficient on older cards.
            jax.jit(lambda x:x*x+1)(jnp.ones(32)).block_until_ready()
            env,qids,vids,targets,changes=prepare_scene(args.planar_proxy,args.scenario)
            model,initial=env.model,env.data;report['model_changes']=changes
            print('Model:',model.nq,'coordinates,',model.neq,'loop constraints; proxy:',args.planar_proxy,flush=True)
            reference,cpu=cpu_rollout(model,initial,qids,vids,targets,args.steps);report['cpu_reference']=cpu
            if not cpu['finite'] or cpu['peak_closure_gap_m']>.002:raise RuntimeError('CPU reference failed finite-state or linkage-closure validation')
            mx=mjx.put_model(model,impl='jax');single=mjx.put_data(model,initial,impl='jax')
            q=jnp.asarray(qids);v=jnp.asarray(vids);target=jnp.asarray(targets)
            def step(data):
                ctrl=jnp.clip(25*(target-data.qpos[q])-4*data.qvel[v],-TORQUE,TORQUE)
                return mjx.step(mx,data.replace(ctrl=ctrl))
            sites=[(model.site(pre+'left_tip').id,model.site(pre+'right_tip').id) for pre in ('','opp_')]
            def loop(data):
                def body(_,carry):
                    data,gap,finite=carry;data=step(data)
                    now=jnp.max(jnp.stack([jnp.linalg.norm(data.site_xpos[a]-data.site_xpos[b]) for a,b in sites]))
                    return data,jnp.maximum(gap,now),finite&jnp.all(jnp.isfinite(data.qpos))&jnp.all(jnp.isfinite(data.qvel))
                return jax.lax.fori_loop(0,args.steps,body,(data,jnp.array(0.,dtype=data.qpos.dtype),jnp.array(True)))
            for count in args.batch_sizes:
                data=jax.tree.map(lambda x:jnp.broadcast_to(x,(count,)+x.shape),single)
                # Distinct trajectories across worlds, with world zero matching the CPU reference.
                vx=int(model.joint('puck_x').dofadr[0])
                data=data.replace(qvel=data.qvel.at[:,vx].add(jnp.arange(count)*.0001))
                run=jax.jit(jax.vmap(loop))
                print('Compiling',count,'worlds (first compilation can take minutes)...',flush=True)
                began=time.perf_counter();result=run(data);jax.block_until_ready(result);compile_s=time.perf_counter()-began
                timings=[]
                for _ in range(args.repeats):
                    began=time.perf_counter();result=run(data);jax.block_until_ready(result);timings.append(time.perf_counter()-began)
                final,gaps,finite=result;seconds=float(np.median(timings))
                qerr=float(np.max(np.abs(np.asarray(final.qpos[0])-reference.qpos)))
                verr=float(np.max(np.abs(np.asarray(final.qvel[0])-reference.qvel)))
                row={'worlds':count,'compile_and_warmup_seconds':compile_s,'median_seconds':seconds,
                     'physics_steps_per_second':count*args.steps/seconds,'control_steps_equivalent_per_second':count*args.steps/seconds/20,
                     'finite_all_steps':bool(np.asarray(finite).all()),'peak_closure_gap_m':float(np.max(np.asarray(gaps))),
                     'world_zero_qpos':np.asarray(final.qpos[0]).tolist(),'cpu_qpos':reference.qpos.tolist(),
                     'world_zero_max_qpos_error_vs_cpu':qerr,'world_zero_max_qvel_error_vs_cpu':verr}
                report['batches'].append(row);print(json.dumps(row,indent=2),flush=True)
                if not row['finite_all_steps'] or row['peak_closure_gap_m']>.002:raise RuntimeError('Finite-state or 2mm linkage closure check failed')
                if qerr>.005 or verr>.1:raise RuntimeError('CPU/MJX agreement check failed (max coordinate error >0.005 or velocity error >0.1); do not treat throughput as validated physics')
                # CPU/MJX agreement is reported and gated for this short trajectory.
                jax.clear_caches()
            report['status']='benchmark_completed_not_training_validated'
    except Exception as error:
        report['status']='failed';report['error']=f'{type(error).__name__}: {error}'
        print(report['error'],flush=True)
        if isinstance(error,NotImplementedError) and not args.planar_proxy:print('The original model includes cylinder/box contacts unsupported by MJX-JAX. Try the explicit --planar-proxy experiment; it changes collision geometry.',flush=True)
    finally:
        if env is not None:env.close()
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
        print('Report:',args.output.resolve(),flush=True)
    return 0 if report.get('status')=='benchmark_completed_not_training_validated' else 1

if __name__=='__main__':raise SystemExit(main())
