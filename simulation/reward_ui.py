"""Local-only reward editor; HTTP threads never mutate the simulator."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import webbrowser

from airhockey_env import REWARD_FIELDS, Settings

# Positive magnitudes: penalty sliders are subtracted by the environment.
CONTROLS={
 'goal_reward':('Opponent goal','Reward',0,100,1),
 'bank_goal_bonus':('Bank shot ending in a goal','Reward',0,20,.5),
 'center_line_reward':('Crossing toward opponent after a hit','Reward',0,20,.5),
 'useful_contact_reward':('Useful new paddle hit','Reward',0,5,.1),
 'defensive_save_reward':('Turn an incoming puck forward','Reward',0,10,.1),
 'chase_weight':('Pursue reachable puck / recover to defense','Shaping',0,10,.1),
 'conceded_goal_penalty':('Goal conceded','Penalty',0,100,1),
 'behind_paddle_penalty_per_second':('Puck behind paddle (per second)','Penalty',0,20,.1),
 'timeout_penalty':('Rally timed out','Penalty',0,30,.5),
 'failure_penalty':('Illegal paddle position / invalid state','Penalty',0,100,1),
 'time_penalty_per_second':('Elapsed time (per second)','Penalty',0,2,.01),
 'effort_penalty_per_second':('Motor effort coefficient (per second)','Penalty',0,2,.01),
 'event_bonus_budget':('Maximum hit/save/crossing bonuses per rally','Limit',0,40,.5),
}


class RewardPanel:
    """Hold a pending reward transaction until the training thread accepts it."""
    def __init__(self,settings,run_dir,open_browser=True):
        self.lock=threading.Lock();self.pending=None;self.revision=0
        self.token=secrets.token_urlsafe(24)
        self.run_dir=Path(run_dir)
        self.active={k:getattr(settings,k) for k in CONTROLS}
        self.metrics={'steps':0,'goals_scored':0,'goals_conceded':0,'rallies':0,'last_outcome':'waiting'}
        self.status='Waiting for the first training batch'
        panel=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass  # Keep routine browser polling out of training logs.
            def send(self,status,payload,content_type='application/json'):
                content=payload.encode() if isinstance(payload,str) else json.dumps(payload,allow_nan=False).encode()
                self.send_response(status)
                self.send_header('Content-Type',content_type)
                self.send_header('Content-Length',str(len(content)))
                self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.end_headers();self.wfile.write(content)
            def local_request(self):
                allowed={f'127.0.0.1:{panel.server.server_port}',f'localhost:{panel.server.server_port}'}
                host=self.headers.get('Host','')
                origin=self.headers.get('Origin')
                return host in allowed and (origin is None or origin in {'http://'+h for h in allowed})
            def do_GET(self):
                if not self.local_request():
                    return self.send(403,{'error':'Local requests only'})
                if self.path=='/':
                    html=Path(__file__).with_name('reward_panel.html').read_text()
                    return self.send(200,html.replace('__PANEL_TOKEN__',panel.token),'text/html; charset=utf-8')
                if self.path=='/api/state':
                    with panel.lock:
                        state={'active':panel.active.copy(),'pending':panel.pending,'controls':CONTROLS,
                               'defaults':{k:getattr(Settings(),k) for k in CONTROLS},
                               'status':panel.status,'metrics':panel.metrics.copy(),'run_dir':str(panel.run_dir)}
                    return self.send(200,state)
                return self.send(404,{'error':'Not found'})
            def do_POST(self):
                if not self.local_request() or self.headers.get('X-Panel-Token')!=panel.token:
                    return self.send(403,{'error':'Unauthorized request'})
                if self.path!='/api/rewards':
                    return self.send(404,{'error':'Not found'})
                try:
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=16384: raise ValueError('Invalid request size')
                    values=json.loads(self.rfile.read(size))['values']
                    if not isinstance(values,dict) or set(values)!=REWARD_FIELDS:
                        raise ValueError('Send all reward fields together')
                    for name,value in values.items():
                        lo,hi=CONTROLS[name][2:4]
                        if isinstance(value,bool) or not isinstance(value,(float,int)) or not lo<=value<=hi:
                            raise ValueError(f'{name} must be between {lo} and {hi}')
                    with panel.lock:
                        panel.revision+=1
                        panel.pending={'revision':panel.revision,'values':values}
                        panel.status='Changes queued: applying at the next training batch'
                    return self.send(200,{'queued':True})
                except (ValueError,TypeError,KeyError) as error:
                    return self.send(400,{'error':str(error)})
        # Loopback only: this is not a publicly exposed remote-control service.
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'
        print(f'Reward sliders: {self.url}')
        if open_browser:
            webbrowser.open(self.url)

    def take_pending(self):
        with self.lock:
            change=self.pending;self.pending=None
            return change

    def applied(self,change,steps):
        with self.lock:
            self.active=change['values'].copy()
            self.status=(f'Applied revision {change["revision"]} at step {steps}' if self.pending is None
                         else 'A newer change is queued for the next batch')
        # Save a reusable override file and a history of when scoring changed.
        (self.run_dir/'reward_settings.json').write_text(json.dumps(change['values'],indent=2))
        with (self.run_dir/'reward_changes.jsonl').open('a') as stream:
            stream.write(json.dumps({'step':steps,**change})+'\n')

    def update_metrics(self,steps,infos):
        with self.lock:
            self.metrics['steps']=steps
            for info in infos:
                if 'episode' in info:
                    self.metrics['rallies']+=1
                    outcome=info.get('outcome','unknown');self.metrics['last_outcome']=outcome
                    self.metrics['goals_scored']+=int(outcome=='goal_scored')
                    self.metrics['goals_conceded']+=int(outcome=='goal_conceded')
                    self.metrics['last_reward_breakdown']=info.get('episode_rewards',{})

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
