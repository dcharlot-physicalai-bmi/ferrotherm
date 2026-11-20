#!/usr/bin/env python3
"""Read-only v3 grading of energy campaign logs (deadline 2026-11-15T21:30Z).
Rules: METHODOLOGY.md v3 @ dbf9653a; harness 777f0417. Per meter, never pooled.
Emits <host>/<meter>-passed.csv (only passes) and <host>/<meter>-refusals.csv."""
import json, csv, os, glob, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDR_P = ["host","attempt","sha","grading","meter","workload","fabric","pairing","j_per_ess","u_j_per_ess","baseline","thrml_j_per_ess","ratio"]
HDR_R = ["host","attempt","sha","grading","meter","utc_start","preflight","energy_phase","host_lock_held","trials_examined","harness_attributable","refusal_reasons"]
PRE_V3 = ("pre-v3 c82c1e96: regrade-v3 refused. rule1: no window_audit and the c82c1e96 metered window is not sampling-only "
          "(scoring/exact law inside window; cannot be removed from raw counter samples); rule3: brackets are per A/B pair, "
          "no settled bracket between paired trials, so no trial has two adjacent brackets of its own")
def occ(s):
    v=[float(x.split(':')[1]) for x in s.split(',') if ':' in x] if s else []
    return (statistics.mean(v), max(v)) if v else (None,None)
def mgate(rep):
    q=rep.get('quiet_gate') or {}
    return q.get('max_load1')
out={}
def add(host,meter,row): out.setdefault((host,meter),[]).append(row)
def reasons_from_trials(trials, inst, scope):
    rs=set(); n=0; att=0
    for t in trials:
        j=t.get('joules') or {}
        if j.get('instrument')!=inst or (scope and j.get('scope')!=scope): continue
        n+=1; att+= bool(j.get('attributable'))
        r=t.get('trial_refusal') or j.get('note') or ''
        if r: rs.add('trial: '+r[:120])
    return n,att,rs
def pre_v3_host(host, d, meters, sha):
    ev=[json.loads(l) for l in open(os.path.join(d,'events.jsonl'))]
    res={e['attempt']:e for e in ev if e.get('event')=='result'}
    for n,e in sorted(res.items()):
        try: rep=json.load(open(os.path.join(d,'attempt-%03d'%n,'report.json')))
        except Exception: rep={}
        ph=rep.get('phases',{}); en=rep.get('energy') or {}
        lock=(rep.get('host_lock') or {}).get('held', e.get('host_lock_held'))
        for meter,(inst,scope) in meters.items():
            rs=[]
            pf=ph.get('preflight',{}).get('grade'); eg=ph.get('energy',{}).get('grade')
            if lock is False: rs.append('lock_not_held')
            if pf=='load_gated': rs.append('load_gated (preflight quiet gate, max load1 %s > 2)'%mgate(rep))
            nt=att=0
            if eg not in (None,'not_run'):
                nt,att,trs=reasons_from_trials(en.get('trials',[]),inst,scope)
                if en.get('outcome') not in ('measured',None) or eg!='quiet':
                    rs.append('energy %s: %s'%(eg,(en.get('note') or '').split(' THRML, when')[0].split(' Grid-energy')[0][:160]))
                rs.extend(sorted(trs))
                rs.append(PRE_V3)
            elif pf!='load_gated' and lock is not False:
                rs.append('energy not_run')
            if lock is None: rs.append('host_lock_held not recorded (pre-lock attempt)')
            add(host,meter,[host,n,sha,'regrade-v3' if eg not in (None,'not_run') else 'pre-v3 (not regradable: energy not run)',meter,e.get('utc_start'),pf,eg,lock,nt,att,' | '.join(rs)])
# M5
pre_v3_host('m5', os.path.join(ROOT,'m5'), {'macmon-soc':('macmon',None)}, 'c82c1e96')
# thor
pre_v3_host('thor.local', os.path.join(ROOT,'thor/thor/evidence-c82c1e9-quiet24-thor'), {'ina238-2-0044-VIN':('ina238',None)}, 'c82c1e96')
# framework
pre_v3_host('framework-128gb-0000', os.path.join(ROOT,'framework/framework/evidence-c82c1e9-quiet24'), {'rapl-package':('rapl','package'),'rapl-core':('rapl','core')}, 'c82c1e96')
# jetson-hub native v3
d=os.path.join(ROOT,'jetson-hub/jetson-hub/campaign-v3-777f041-jetsonhub')
ev=[json.loads(l) for l in open(os.path.join(d,'events.jsonl'))]
starts={e['attempt']:e for e in ev if e.get('event')=='start'}
res={e['attempt']:e for e in ev if e.get('event')=='result'}
JM={'rapl-package':('rapl','package'),'rapl-core':('rapl','core'),'nvml-gpu':('nvml','gpu')}
for n,s in sorted(starts.items()):
    e=res.get(n)
    for meter,(inst,scope) in JM.items():
        if e is None:
            add('jetson-hub',meter,['jetson-hub',n,'777f0417','native-v3',meter,s.get('utc_start'),'quiet (log)','incomplete','held (log)',0,0,
              'unbracketed/ungraded: attempt never wrote report.json; THRML arm failed to start in-window throughout (JAX CPU device absent in the -cuda venv, CUDA_ERROR_NO_DEVICE, THRML_LIMIT oom SIGKILL); log last written 2026-11-14T15:21Z; driver gone after host reboot 2026-11-16 15:51 ET; no window_audit, no graded brackets'])
            continue
        rep=json.load(open(os.path.join(d,'attempt-%03d'%n,'report.json')))
        ph=rep['phases']; pf=ph['preflight']['grade']; eg=ph['energy']['grade']; lock=e.get('host_lock_held')
        rs=[]; nt=att=0
        if lock is not True: rs.append('lock_not_held')
        if pf=='load_gated': rs.append('load_gated (preflight quiet gate, max load1 %s > 2)'%mgate(rep))
        if eg not in ('not_run',None):
            for half in ('suite','grid'):
                h=rep['energy'].get(half) or {}
                a,b,trs=reasons_from_trials(h.get('trials',[]),inst,scope); nt+=a; att+=b
                if h.get('outcome')!='measured': rs.append('%s %s: %s'%(half,h.get('outcome'),(h.get('note') or '').split(' Grid-energy')[0][:200]))
                for t in h.get('trials',[]):
                    j=t.get('joules') or {}
                    if j.get('instrument')!=inst or j.get('scope')!=scope: continue
                    m,mx=occ(t.get('other_core_busy'))
                    if j.get('attributable') and m is not None and (m>0.05 or mx>0.15):
                        rs.append('trial %s/%s: harness marked attributable but trial-window other-core rule fails (other-core mean %.3f > 0.05; core max %.3f > 0.15) - the rule is per trial, not per meter'%(t['workload'],t.get('fabric'),m,mx))
                    if t.get('paired_delta_j') is None: rs.append('trial %s/%s: no adjacent THRML A or B trial (ABBA incomplete; next idle bracket refused) - no paired comparison'%(t['workload'],t.get('fabric')))
                rs.extend(sorted(trs))
                for sm in h.get('summaries',[]):
                    if sm.get('instrument')==inst and sm.get('scope')==scope and sm.get('joules_per_ess') is None:
                        rs.append('summary %s/%s: J/ESS null (n=%s)'%(sm['workload'],sm.get('fabric'),sm.get('n')))
        add('jetson-hub',meter,['jetson-hub',n,'777f0417','native-v3',meter,s.get('utc_start'),pf,eg,lock,nt,att,' | '.join(dict.fromkeys(rs))])
summary={}
for (host,meter),rows in sorted(out.items()):
    hd=os.path.join(ROOT,'receipt',host); os.makedirs(hd,exist_ok=True)
    with open(os.path.join(hd,meter+'-refusals.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(HDR_R); w.writerows(rows)
    with open(os.path.join(hd,meter+'-passed.csv'),'w',newline='') as f:
        csv.writer(f).writerow(HDR_P)
    summary['%s/%s'%(host,meter)]={'attempts_examined':len(rows),'passed':0,'refused':len(rows),
        'energy_phase_attempts':sum(1 for r in rows if r[7] not in (None,'not_run')),'trials_examined':sum(r[9] for r in rows),'harness_attributable_trials':sum(r[10] for r in rows)}
json.dump(summary,open(os.path.join(ROOT,'receipt','summary.json'),'w'),indent=1)
print(json.dumps(summary,indent=1))
