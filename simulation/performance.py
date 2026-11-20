"""Portable CLI performance report, with Linux CPU sampling and NVIDIA telemetry."""
import json,subprocess,time
from pathlib import Path

def report(interval=1.):
    result={'cpu_percent':None,'memory_used_mib':None,'memory_total_mib':None,'gpus':[]}
    def cpu():
        values=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(values[:8]),values[3]+values[4]
    try:
        a,idle=cpu();time.sleep(interval);b,idle2=cpu();result['cpu_percent']=round(100*(1-(idle2-idle)/max(1,b-a)),1)
        mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        result['memory_total_mib']=round(mem['MemTotal']/1024);result['memory_used_mib']=round((mem['MemTotal']-mem['MemAvailable'])/1024)
    except (OSError,ValueError,KeyError):
        try:
            import psutil
            result['cpu_percent']=psutil.cpu_percent(interval=interval);m=psutil.virtual_memory();result['memory_total_mib']=round(m.total/1048576);result['memory_used_mib']=round(m.used/1048576)
        except ImportError:result['cpu_note']='CPU utilization requires Linux /proc or optional psutil'
    try:
        out=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
        for line in out.stdout.splitlines():
            fields=[x.strip() for x in line.split(',')]
            if len(fields)==6:result['gpus'].append(dict(zip(['name','utilization_percent','memory_used_mib','memory_total_mib','temperature_c','power_w'],fields)))
    except (OSError,subprocess.SubprocessError) as exc:result['gpu_note']=str(exc)
    return result

def main():print(json.dumps(report(),indent=2))
if __name__=='__main__':main()
