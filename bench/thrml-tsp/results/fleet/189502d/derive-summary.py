# Reproduce per-workload-summary.json from unmodified serialized trial records.
import json,pathlib,math
root=pathlib.Path(__file__).parent
out=[]
def ci(xs):
 assert len(xs)==5 and all(x is not None and math.isfinite(x) for x in xs)
 mean=sum(xs)/5;var=sum((x-mean)**2 for x in xs)/4;half=2.7764451052*math.sqrt(var/5)
 return dict(mean=mean,lo95=mean-half,hi95=mean+half,n=5)
for host in ["framework","m5"]:
 d=json.loads((root/host/"evidence-189502d/contended/report.json").read_text());e=d["energy"];assert len(e["trials"])==40
 for wi,w in enumerate(d["workloads"]):
  block=e["trials"][wi*10:(wi+1)*10]
  for i,t in enumerate(block):assert (t["side"],t["trial"])==("ferrotherm" if i%2==0 else "thrml",i//2)
  for side in ["ferrotherm","thrml"]:
   ts=[t for t in block if t["side"]==side]
   out.append(dict(host=host,workload=w["id"],side=side,evidence_grade=e["evidence_grade"],rank=ts[0]["rank"],gross_j=ci([t["joules"]["gross_j"] for t in ts]),incremental_j=ci([t["joules"]["incremental_j"] for t in ts]),joules_per_ess=ci([t["joules_per_ess"] for t in ts]),load1_min=min(t["load1"] for t in ts),load1_max=max(t["load1"] for t in ts),duration_min_s=min(t["joules"]["seconds"] for t in ts),duration_max_s=max(t["joules"]["seconds"] for t in ts)))
(root/"per-workload-summary.json").write_text(json.dumps(out,indent=2)+"\n")
