from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path.cwd(); research=root/'research/voice-prosody'; phase=sys.argv[1]
assert phase in ('predict','acquire-ref','score')
folder=research/'runs/hybrid-validation-execution-v1';folder.mkdir(exist_ok=True)
record=folder/(phase+'.json');log=folder/(phase+'.log')
if record.exists() or log.exists(): raise FileExistsError('No overwrite of execution evidence')
command=[str(research/'.venv/bin/python'),'-m','phenometrix_prosody.hybrid_validation',phase,
 '--freeze','runs/hybrid-validation-freeze-v1/freeze.json','--output','runs/hybrid-validation-v1']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
m={'schema':'phenometrix-local-serialized-natural-phase.v1','phase':phase,'started_at':now(),
 'command':command,'cwd':str(research),'compute':'local','runner_sha256':sha(Path(__file__)),
 'freeze_sha256':sha(research/'runs/hybrid-validation-freeze-v1/freeze.json'),'status':'running'}
record.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m),flush=True)
with log.open('x') as out:r=subprocess.run(command,cwd=research,stdout=out,stderr=subprocess.STDOUT)
m.update(status='complete' if r.returncode==0 else 'failed',exit_code=r.returncode,finished_at=now(),log_sha256=sha(log))
record.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m),flush=True);sys.exit(r.returncode)
