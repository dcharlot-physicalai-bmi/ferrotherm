// Drive the DEPLOYED editor, not a local build. A page that loads is not a page that is current.
import { chromium } from 'playwright';
const URL = 'https://energy.physicalai-bmi.org/assets/ferrotherm/graph.html';
const b = await chromium.launch(); const p = await b.newPage();
const errs = [];
p.on('pageerror', e => errs.push('pageerror: ' + e.message));
p.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
await p.goto(URL, { waitUntil: 'load' });
await p.evaluate(() => window.ferrotherm.ready);

const solved = await p.evaluate(() => {
  const F = window.ferrotherm;
  F.clear();
  const vs = []; for (let i = 0; i < 4; i++) vs.push(F.add('binary', 40, 40+i*90, {name:'v'+i}));
  const c = F.add('atmost', 320, 60, { k: 2, value: 1 });
  ['a','b','c','d'].forEach((pin,i) => F.connect(vs[i], c, pin));
  const s = F.add('solve', 620, 200); F.connect(c, s);
  vs.forEach((v,i) => { const o = F.add('prefer',320,360+i*80,{value:1,weight:4-i,maximize:1});
                        F.connect(v,o,'var'); F.connect(o,s); });
  F.connect(s, F.add('report', 900, 200), 'result');
  return F.run();
});
const on = [...solved.matchAll(/v(\d) = 1/g)].map(m => 'v'+m[1]);
console.log('at most 2 of four, weights 4..1 ->', JSON.stringify(on),
            solved.includes('feasible  yes') ? '| feasible' : '| INFEASIBLE');
console.log('objective terms:', solved.match(/(\d+) objective terms/)?.[1]);

const refusal = await p.evaluate(() => {
  const F = window.ferrotherm;
  F.clear();
  const t = F.add('integer', 40, 40, { name: 'temperature', lo: 10, hi: 20 });
  const fx = F.add('fix', 300, 40, { value: 3 });
  F.connect(t, fx, 'var');
  const s = F.add('solve', 600, 40); F.connect(fx, s);
  F.connect(s, F.add('report', 860, 40), 'result');
  return F.run();
});
console.log('refusal:', refusal.split('\n')[0].slice(0, 92));
console.log(errs.length ? 'ERRORS: ' + errs.join(' | ') : 'no page errors');
await b.close();
