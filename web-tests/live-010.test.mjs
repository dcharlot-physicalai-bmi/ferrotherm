// Drive the DEPLOYED workbench through everything 0.10.0 added. The wasm and the page are separate
// fetches, so byte-identical to the repo is not the same as running.
import { chromium } from 'playwright';
const URL = 'https://energy.physicalai-bmi.org/assets/ferrotherm/ide.html?cb=' + Math.random();
const b = await chromium.launch(); const p = await b.newPage();
const errs = [];
p.on('pageerror', e => errs.push('pageerror: ' + e.message));
p.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
await p.goto(URL, { waitUntil: 'load' });
await p.waitForFunction(() => document.getElementById('status')?.textContent
  && !document.getElementById('status').textContent.includes('loading'), null, { timeout: 30000 });

let fail = 0;
const check = (n, ok, d = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${n}${d ? '   ' + d : ''}`); if (!ok) fail++; };
const run = (json) => p.evaluate((j) => {
  const el = document.getElementById('src');
  el.value = j; el.dispatchEvent(new Event('input', { bubbles: true }));
  if (typeof window.apply === 'function') window.apply(); else document.getElementById('apply')?.click();
  return { status: document.getElementById('status').textContent,
           hint: document.getElementById('viewhint').textContent };
}, json);

const latin = await run(JSON.stringify({
  variables: [0,1,2,3].map(i => ({ name: `c${i}`, values: 4 })),
  constraints: [{ type: 'all_different', of: [0,1,2,3].map(i => ({ var: `c${i}` })) }], tries: 60 }));
check('all_different solves a latin square row', /every constraint holds/.test(latin.status), latin.status.slice(0,70));

const pigeon = await run(JSON.stringify({
  variables: [0,1,2,3,4].map(i => ({ name: `x${i}`, values: 3 })),
  constraints: [{ type: 'all_different', of: [0,1,2,3,4].map(i => ({ var: `x${i}` })) }] }));
check('pigeonhole refused at compile time', /No assignment can satisfy/.test(pigeon.status), pigeon.status.slice(0,80));

const bin = await run(JSON.stringify({ variables: [{ name: 'x', values: 6, encoding: 'binary' }], tries: 4 }));
check('binary encoding honoured (3 spins, not 6)', /1 variables over 3 spins/.test(bin.hint), bin.hint);
check('and its caveat outranks "every constraint holds"', /caveat/i.test(bin.status), bin.status.slice(0,90));

const bad = await run(JSON.stringify({ variables: [{ name: 'x', values: 6, encoding: 'gray' }] }));
check('unknown encoding refused by listing the real ones', /unknown encoding/.test(bad.status), bad.status.slice(0,70));

const soft = await run(JSON.stringify({
  variables: [{ name: 'a', values: 2 }, { name: 'b', values: 2 }],
  constraints: [{ type: 'not_equal', a: 'a', b: 'b', soft: 1 }],
  objective: { maximize: true, terms: [{ var: 'a', value: 0, weight: 5 }, { var: 'b', value: 0, weight: 5 }] }, tries: 24 }));
check('a priced preference is traded', /preference\(s\) traded/.test(soft.status), soft.status.slice(0,80));

check('no page errors', errs.length === 0, errs.join(' | '));
await b.close();
console.log(fail ? `\n${fail} failed` : '\nall passed against the deployed 0.10.0 page');
process.exit(fail ? 1 : 0);
