"""Stage timing and independent display scheduling for the camera tracker."""
import time
from collections import defaultdict


class FrameProfiler:
    def __init__(self,enabled=False,interval=3.0,clock=time.perf_counter):
        self.enabled=enabled;self.interval=interval;self.clock=clock
        self.start=self.clock();self.last=self.start;self.frames=0;self.sums=defaultdict(float)
    def begin(self):
        if self.enabled:self.last=self.clock()
    def mark(self,name):
        if self.enabled:
            now=self.clock();self.sums[name]+=now-self.last;self.last=now
    def end(self):
        if not self.enabled:return
        self.mark('keys_other');self.frames+=1
        now=self.clock();elapsed=now-self.start
        if elapsed>=self.interval:
            timings=' | '.join(f'{k} {1000*v/self.frames:.2f} ms' for k,v in self.sums.items())
            print(f'PROFILE: {self.frames/elapsed:.1f} loop FPS | {timings}',flush=True)
            self.start=now;self.frames=0;self.sums.clear()


class DisplaySchedule:
    def __init__(self,fps):self.interval=1/fps if fps>0 else 0;self.next=0.0
    def due(self,now):
        if now<self.next:return False
        self.next=now+self.interval
        return True
