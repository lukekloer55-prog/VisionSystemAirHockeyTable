from benchmark_mjx import prepare_scene,cpu_rollout
import numpy as np
for scenario in ['rail','paddle']:
 results=[]
 for proxy in [False,True]:
  e,q,v,t,changes=prepare_scene(proxy,scenario)
  final,report=cpu_rollout(e.model,e.data,q,v,t,200)
  assert report['finite'] and report['peak_closure_gap_m']<.002 and report['steps_with_contact']>0
  results.append(final.qpos.copy());e.close()
 print(scenario,'proxy difference',np.max(np.abs(results[0]-results[1])))
 assert np.max(np.abs(results[0]-results[1]))<.005
print('PASS: CPU planar proxy bounce/contact and closure checks')
