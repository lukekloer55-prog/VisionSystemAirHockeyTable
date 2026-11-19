"""Measured PlotClock linkage demo. Simulation only: no hardware commands.

Run: uv run python plotclock_demo.py
Check: uv run python plotclock_demo.py --check
"""
import argparse
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from table_demo import build_scene, LENGTH

# All dimensions are metres; joint angles are radians; actuator forces are N*m.
GEOMETRY = json.loads(Path(__file__).with_name('plotclock_geometry.json').read_text())
P = {name: np.array(value, dtype=float) for name, value in GEOMETRY['pivots'].items()}
TORQUE = GEOMETRY['motor_torque_limit_nm']
# Place the servo axes behind the nearest base face, using confirmed mounting dimensions.
BASE_SETBACK = GEOMETRY["base_face_setback_m"] + GEOMETRY["base_pivot_offset_from_near_face_m"]
BASE_Y = -LENGTH / 2 - BASE_SETBACK
ARM_Z = 0.10  # Elevated planar mechanism; actual mounting height is not supplied.


def angle(vector):
    """Find a vector's direction in the table plane."""
    return math.atan2(vector[1], vector[0])


def wrap(value):
    """Use the shortest angular difference instead of jumping by a full turn."""
    return math.atan2(math.sin(value), math.cos(value))


def chain(side):
    """Recover both link lengths and initial angles from the CAD pivot centers."""
    base, elbow = P[side + '_base'], P[side + '_elbow']
    upper, lower = elbow - base, P['tip'] - elbow
    return np.linalg.norm(upper), np.linalg.norm(lower), angle(upper), wrap(angle(lower)-angle(upper))


def motor_targets(target):
    """Solve each two-link chain and retain the elbow branch used in the CAD pose."""
    values = []
    for side in ('left', 'right'):
        length1, length2, home, _ = chain(side)
        delta = np.asarray(target)-P[side+'_base']
        distance = np.linalg.norm(delta)
        if distance < 1e-9 or not abs(length1-length2) < distance < length1+length2:
            raise ValueError('Target outside the nonsingular linkage workspace.')
        cosine = (length1**2+distance**2-length2**2)/(2*length1*distance)
        offset = math.acos(float(np.clip(cosine,-1,1)))
        # Pick the branch closest to the original servo angle.
        candidates = [wrap(angle(delta)+sign*offset-home) for sign in (-1,1)]
        joint_target = min(candidates,key=abs)
        if abs(joint_target) > 0.35:
            raise ValueError('Target exceeds the provisional +/-0.35 rad base-joint range.')
        values.append(joint_target)
    return np.array(values)


def make_model():
    """Build two open chains and close their tips with an equality constraint."""
    root = ET.fromstring(build_scene())
    root.set('model','Measured PlotClock direct-drive demo')
    root.find('option').set('integrator','implicitfast')
    root.find('option').set('iterations','100')
    world = root.find('worldbody')
    # The existing table remains as context. Remove the puck during arm commissioning.
    world.remove(world.find("body[@name='puck']"))
    mass = GEOMETRY['assumed_link_mass_kg']
    width, thick = GEOMETRY['link_width'], GEOMETRY['link_thickness']
    ET.SubElement(world,'geom',name='fixed_base',type='box',pos=f'0 {BASE_Y} {ARM_Z}',
                  size=f'{np.linalg.norm(P["right_base"]-P["left_base"])/2} {width/2} {thick/2}',
                  rgba='.25 .28 .32 1',contype='0',conaffinity='0')
    actuator = ET.SubElement(root,'actuator')
    for side, color in [('left','.15 .45 .9 1'),('right','.95 .45 .1 1')]:
        length1,length2,home,elbow_angle = chain(side)
        base = P[side+'_base']
        # Each hinge's zero angle is the assembled CAD pose, not a straight arm.
        upper = ET.SubElement(world,'body',name=side+'_upper',
            pos=f'{base[0]} {BASE_Y+base[1]} {ARM_Z}',euler=f'0 0 {home}')
        ET.SubElement(upper,'joint',name=side+'_motor',type='hinge',axis='0 0 1',
            limited='true',range='-.35 .35',damping='.08',
            armature=str(GEOMETRY['assumed_rotor_inertia_kg_m2']))
        ET.SubElement(upper,'geom',type='box',pos=f'{length1/2} 0 0',
            size=f'{length1/2} {width/2} {thick/2}',mass=str(mass),rgba=color,
            contype='0',conaffinity='0')
        lower = ET.SubElement(upper,'body',name=side+'_lower',
            pos=f'{length1} 0 0',euler=f'0 0 {elbow_angle}')
        ET.SubElement(lower,'joint',name=side+'_elbow',type='hinge',axis='0 0 1',damping='.02')
        # Stack distal links visually. Joint axes and closure remain in one plane.
        layer = thick if side == 'left' else -thick
        ET.SubElement(lower,'geom',type='box',pos=f'{length2/2} 0 {layer}',
            size=f'{length2/2} {width/2} {thick/2}',mass=str(mass),rgba=color,
            contype='0',conaffinity='0')
        ET.SubElement(lower,'site',name=side+'_tip',pos=f'{length2} 0 0',size='.008',rgba='0 1 0 1')
        if side == 'left':
            # Visual marker for the controlled pin, not an assumed paddle collider.
            ET.SubElement(lower,'geom',name='end_pin',type='cylinder',pos=f'{length2} 0 0',
                size='.00633239626845922 .0381',mass='.02',rgba='.2 .8 .3 1',contype='0',conaffinity='0')
        ET.SubElement(actuator,'motor',name=side+'_drive',joint=side+'_motor',gear='1',
            ctrllimited='true',ctrlrange=f'{-TORQUE} {TORQUE}',forcelimited='true',
            forcerange=f'{-TORQUE} {TORQUE}')
    equality = ET.SubElement(root,'equality')
    # Two coincident tip sites close the five-bar mechanism; passive hinges stay free.
    ET.SubElement(equality,'connect',name='tip_connection',site1='left_tip',site2='right_tip',
                  solref='.003 1',solimp='.99 .999 .001')
    ET.SubElement(world,'body',name='target',mocap='true',pos=f'{P["tip"][0]} {BASE_Y+P["tip"][1]} {ARM_Z}',)
    ET.SubElement(world.find("body[@name='target']"),'site',name='target_marker',type='sphere',
                  size='.012',rgba='1 0 .8 .6')
    # Euler values above are radians, matching all Python kinematics.
    ET.SubElement(root,'compiler',angle='radian')
    ET.indent(root, space='  ')
    xml = ET.tostring(root,encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    return model,data,xml


def drive(model,data,moving=True):
    """Follow a small 20 mm circle with a torque-limited simulated position loop."""
    phase = 2*math.pi*data.time/8  # Eight seconds per circuit; no high-speed motion.
    target = P['tip'] + (np.array([.02*(math.cos(phase)-1),.02*math.sin(phase)]) if moving else 0)
    desired = motor_targets(target)
    # The small, slow target path does not use the motor rated RPM as a command.
    for i,side in enumerate(('left','right')):
        joint = data.joint(side+'_motor')
        torque = 25*(desired[i]-joint.qpos[0])-4*joint.qvel[0]
        data.ctrl[i] = np.clip(torque,-TORQUE,TORQUE)
    data.mocap_pos[0] = [target[0],BASE_Y+target[1],ARM_Z]
    return target


def check(model,data):
    """Test a full motion cycle, closure, tracking error, and actual actuator torque."""
    worst_gap=worst_error=peak_torque=0.0
    start=data.site('left_tip').xpos.copy()
    travel=0.0
    for _ in range(12000):
        target=drive(model,data)
        mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)
        tip=data.site('left_tip').xpos
        gap=np.linalg.norm(tip-data.site('right_tip').xpos)
        error=np.linalg.norm(tip[:2]-[target[0],BASE_Y+target[1]])
        worst_gap=max(worst_gap,gap);worst_error=max(worst_error,error)
        peak_torque=max(peak_torque,float(np.max(np.abs(data.actuator_force))))
        travel=max(travel,np.linalg.norm(tip-start))
        assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)), 'Nonfinite state'
    assert worst_gap < .0005, f'Closed linkage separated: {worst_gap} m'
    assert worst_error < .02, f'Excessive tracking error: {worst_error} m'
    assert peak_torque <= TORQUE+1e-9, 'Actuator exceeded torque cap'
    assert travel > .02, 'Arm did not move'
    assert not any(w.number for w in data.warning), 'MuJoCo reported a physics warning'
    print(f'PASS: 12 s rollout; closure gap {worst_gap*1000:.4f} mm; '
          f'max tracking error {worst_error*1000:.2f} mm; peak torque {peak_torque:.3f} N*m.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Run headless mechanism checks')
    parser.add_argument('--hold',action='store_true',help='Hold the initial CAD pose')
    parser.add_argument('--export-xml',type=Path,help='Save the generated model for inspection')
    args=parser.parse_args()
    model,data,xml=make_model()
    if args.export_xml:
        args.export_xml.write_text(xml)
    if args.check:
        check(model,data)
        return
    import mujoco.viewer
    print('SIMULATION ONLY. Blue/orange: arms. Green: pin. Magenta: target.')
    print('Provisional 0.4 kg/link, rotor inertia, gains and joint travel. No paddle collisions.')
    with mujoco.viewer.launch_passive(model,data) as viewer:
        viewer.cam.lookat[:]=[0,-.45,.08]
        viewer.cam.distance=2.3
        viewer.cam.azimuth=90
        viewer.cam.elevation=-65
        while viewer.is_running():
            start=time.monotonic()
            with viewer.lock():
                drive(model,data,not args.hold)
                mujoco.mj_step(model,data)
            viewer.sync()
            time.sleep(max(0,model.opt.timestep-(time.monotonic()-start)))


if __name__=='__main__':
    main()

