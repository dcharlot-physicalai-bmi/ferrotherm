import os,json,time,pathlib,subprocess,datetime
r=pathlib.Path.home()/"bench-scratch";repo=r/"ferrotherm-thrml";dest=r/"evidence-0029c27";dest.mkdir(exist_ok=True)
domains=[pathlib.Path("/sys/class/powercap/intel-rapl:0"),pathlib.Path("/sys/class/powercap/intel-rapl:0:0")]
meta=[{"path":str(d),"name":(d/"name").read_text().strip(),"max_energy_range_uj":int((d/"max_energy_range_uj").read_text())} for d in domains]
(dest/"rapl-domains.json").write_text(json.dumps(meta,indent=2))
trace=open(dest/"rapl-raw.jsonl","w",buffering=1);events=open(dest/"events.jsonl","w",buffering=1);prev={}
def sample(mode):
 row={"timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),"monotonic":time.monotonic(),"mode":mode,"loadavg":os.getloadavg()}
 for d in meta:
  n=d["name"];v=int((pathlib.Path(d["path"])/"energy_uj").read_text());row[n+"_uj"]=v
  if n in prev:row[n+"_delta_uj"]=(v-prev[n])%d["max_energy_range_uj"];row[n+"_wrapped"]=v<prev[n]
  prev[n]=v
 trace.write(json.dumps(row)+"\n")
def run(mode):
 out=dest/mode;cmd=[str(repo/"target/release/ferrotherm-thrml-bench"),"--out",str(out),"--trials","5","--python",str(r/"thrml-venv/bin/python")]
 if mode=="contended":cmd+=["--contended"]
 events.write(json.dumps({"mode":mode,"command":cmd,"start":datetime.datetime.now(datetime.timezone.utc).isoformat()})+"\n");print(mode,"start",flush=True)
 with open(dest/(mode+".log"),"w") as log:
  p=subprocess.Popen(cmd,cwd=repo,env=dict(os.environ,JAX_PLATFORMS="cpu"),stdout=log,stderr=subprocess.STDOUT)
  while p.poll() is None:sample(mode);time.sleep(.1)
 events.write(json.dumps({"mode":mode,"exit_code":p.returncode,"end":datetime.datetime.now(datetime.timezone.utc).isoformat()})+"\n");print(mode,"exit",p.returncode,flush=True)
 if p.returncode:raise RuntimeError("harness failed")
 d=json.loads((out/"report.json").read_text());print(mode,"energy",d["energy"]["outcome"],d["energy"]["evidence_grade"],flush=True);return d
try:
 d=run("quiet")
 if d["energy"]["evidence_grade"]=="load_gated":run("contended")
 (dest/"exit.txt").write_text("0\n")
finally:trace.close();events.close()
