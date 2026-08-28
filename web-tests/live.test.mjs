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
// The vocabulary this page could not state until it could: one all_different rather than three
// pairwise inequalities, and a constraint priced as a preference rather than imposed as a rule.
const said = await p.evaluate(() => {
  const F = window.ferrotherm;
  F.clear();
  const vs = ['west', 'middle', 'east'].map((n, i) =>
    F.add('categorical', 40, 40 + i * 100, { name: n, values: 3 }));
  const ad = F.add('alldifferent', 320, 90);
  vs.forEach(v => F.connect(v, ad));
  const s = F.add('solve', 620, 90); F.connect(ad, s);
  F.connect(s, F.add('report', 900, 90), 'result');
  const txt = F.run();
  return { got: [...txt.matchAll(/(west|middle|east) = (\d)/g)].map(m => m[2]),
           feasible: txt.includes('feasible  yes') };
});
console.log('all_different ->', JSON.stringify(said.got),
            new Set(said.got).size === 3 ? '| all distinct' : '| NOT DISTINCT',
            said.feasible ? '| feasible' : '| INFEASIBLE');

const traded = await p.evaluate(() => {
  const F = window.ferrotherm;
  F.clear();
  const a = F.add('binary', 40, 40, { name: 'a' }), b = F.add('binary', 40, 160, { name: 'b' });
  const ne = F.add('notequal', 300, 60, { soft: 1 });
  F.connect(a, ne, 'a'); F.connect(b, ne, 'b');
  const s = F.add('solve', 620, 60); F.connect(ne, s);
  for (const v of [a, b]) {
    const o = F.add('prefer', 320, 300 + v * 60, { value: 1, weight: 20, maximize: 1 });
    F.connect(v, o, 'var'); F.connect(o, s);
  }
  F.connect(s, F.add('report', 900, 60), 'result');
  const txt = F.run();
  return { traded: /traded:/.test(txt), broken: /broken:/.test(txt),
           feasible: txt.includes('feasible  yes') };
});
console.log('a priced preference ->', traded.traded ? 'traded' : 'NOT TRADED',
            traded.broken ? '| reported as BROKEN' : '| not called broken',
            traded.feasible ? '| still feasible' : '| INFEASIBLE');

// The round trip, on the deployed bytes: the model an agent writes, the picture, and back.
const trip = await p.evaluate(() => {
  const F = window.ferrotherm;
  const m = { variables: [{ name: 'x', values: 3 }, { name: 'y', values: 3 }],
              constraints: [{ type: 'not_equal', a: 'x', b: 'y', soft: 2 }],
              objective: { maximize: true, terms: [{ var: 'x', value: 2, weight: 5 }] } };
  F.fromModel(m);
  const back = F.toModel();
  return { link: F.link().length, soft: back.constraints[0]?.soft,
           weight: back.objective?.terms[0]?.weight };
});
console.log('round trip -> soft', trip.soft, '| weight', trip.weight,
            '| link', trip.link, 'chars');

console.log(errs.length ? 'ERRORS: ' + errs.join(' | ') : 'no page errors');
await b.close();
