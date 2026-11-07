import os,json,time,pathlib,subprocess,datetime,signal
r=pathlib.Path.home()/"bench-scratch";repo=r/"ferrotherm-thrml";dest=r/"evidence-8e9ad0a-scale";dest.mkdir(exist_ok=True)
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
 out=dest/mode;cmd=[str(repo/"target/release/ferrotherm-thrml-bench"),"--out",str(out),"--trials","5","--scale","--device","cpu","--python",str(r/"thrml-venv/bin/python")]
 if mode=="contended":cmd+=["--contended"]
 events.write(json.dumps({"mode":mode,"command":cmd,"start":datetime.datetime.now(datetime.timezone.utc).isoformat()})+"\n");print(mode,"start",flush=True)
 with open(dest/(mode+".log"),"w") as log:
  p=subprocess.Popen(cmd,cwd=repo,env=dict(os.environ,JAX_PLATFORMS="cpu"),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  last_progress=time.monotonic();last_size=0;started=last_progress;timed_out=False
  while p.poll() is None:
   sample(mode);size=log.tell()
   if size!=last_size:last_size=size;last_progress=time.monotonic()
   if time.monotonic()-last_progress>180 or time.monotonic()-started>1800:
    timed_out=True
    text=(dest/(mode+".log")).read_text(errors="replace").splitlines()
    record={"mode":mode,"status":"timed-out","no_progress_limit_s":180,"whole_suite_limit_s":1800,"elapsed_s":time.monotonic()-started,"last_log_lines":text[-12:],"note":"Harness has no per-workload selector or timeout. Interrupted only our process group; downstream cells are not-run, not measured."}
    (dest/(mode+"-timeout.json")).write_text(json.dumps(record,indent=2))
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    break
   time.sleep(.1)
 events.write(json.dumps({"mode":mode,"exit_code":p.returncode,"end":datetime.datetime.now(datetime.timezone.utc).isoformat()})+"\n");print(mode,"exit",p.returncode,flush=True)
 if timed_out:return None
 if p.returncode:raise RuntimeError("harness failed")
 d=json.loads((out/"report.json").read_text());print(mode,"energy",d["energy"]["outcome"],d["energy"]["evidence_grade"],flush=True);return d
try:
 d=run("quiet")
 if d is not None and d["energy"]["evidence_grade"]=="load_gated":d=run("contended")
 (dest/"exit.txt").write_text("124\n" if d is None else "0\n")
finally:trace.close();events.close()
