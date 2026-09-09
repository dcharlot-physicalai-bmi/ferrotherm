#!/usr/bin/env python3
"""A quantitative scorecard for this crate: consistency, documentation, test rigour, ease of use.

  scripts/quality.py            # print the scorecard, exit non-zero on an unexplained gap
  scripts/quality.py --report   # print it and always exit 0

WHY THIS EXISTS. The crate had 21 gates saying "is it correct" and none saying "is it CONSISTENT",
"is it documented" or "is it usable". Those are the properties a reader meets first and no test can
fail on. They are also the ones that rot silently: a module added without tests, an error returned
as a String where its neighbours return an enum, a doc block written as ```text so nothing compiles
it -- each is invisible to a green suite.

MEASURED, NOT JUDGED. Every number here is a count with a stated definition, so it can be re-run and
compared. Nothing weighs the numbers into a score: a single number would hide which of its parts
moved, and the parts are the actionable thing.

THE EXEMPTION TABLE IS THE AUDIT, AND THE RATCHET IS THE REST OF IT. A module with public API and
no TESTS must carry a written reason in REASONS below -- writing the sentence is what forces the
question, and a reason that stops being true is visible in a way an absent test never is.

Mutation coverage is a ratchet instead, and that is a deliberate choice rather than a softer bar.
When this script was first run, 47 of 80 public modules had no recorded mutation: the suite this
crate leans on covers a third of it. Failing on all 47 would make the gate permanently red, which is
how a gate gets ignored; writing 47 exemption sentences would fill the table with "not done yet",
which is not a reason and would corrupt the one mechanism that works. So the COUNT is pinned and may
only fall. A new module without a mutation row fails immediately; the existing debt is visible in
every run and shrinks on purpose.

Read with Python rather than shelling out to grep: this machine's `grep` is an embedded ugrep that
silently skips gitignored files, and a metric that quietly measures a subset is worse than none.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

# Modules allowed to carry no #[test] and/or no recorded mutation, each with the reason.
# A module here that has since grown tests is reported as STALE -- an exemption nobody removed is an
# exemption nobody re-read.
# Modules with public API and no recorded mutation, at the last time this was pinned. It may only
# go DOWN: a new module without a mutation row pushes the count over the line and fails. Lower it
# whenever you add rows -- leaving it high after doing the work turns the ratchet into a ceiling.
MAX_UNMUTATED = 47

REASONS = {
    "lib": "the crate root: module declarations and the crate doc, no logic to test",
    "wgsl": "GPU shader source as string constants; compiled by `ferrotherm-gpu`'s own tests against a real adapter, which this crate cannot do without one",
    "targets": "a table of device descriptions; every field is data and `ledger` tests the arithmetic that reads it",
}

def modules():
    out = {}
    for f in sorted(os.listdir(SRC)):
        if f.endswith(".rs"):
            with open(os.path.join(SRC, f), encoding="utf-8") as fh:
                out[f[:-3]] = fh.read()
    return out

def public_fns(text):
    """`pub fn` and `pub const fn` declarations, with their parameter lists."""
    return re.findall(r"^\s*pub (?:const )?(?:unsafe )?(?:extern \"C\" )?fn\s+(\w+)\s*(\([^{;]*)", text, re.M)

def strip_tests(text):
    """Everything before `#[cfg(test)]`, so test code never counts as API or as documentation."""
    i = text.find("#[cfg(test)]")
    return text if i < 0 else text[:i]

def main():
    report_only = "--report" in sys.argv
    mods = modules()
    mut_path = os.path.join(ROOT, "scripts", "mutation-suite.sh")
    mut_text = open(mut_path, encoding="utf-8").read() if os.path.exists(mut_path) else ""
    mutated = set(re.findall(r'"src/(\w+)\.rs\|', mut_text))

    rows, failures, stale = [], [], []
    tot = dict(pub_fns=0, result=0, option=0, must_use=0, wide=0, seeded=0,
               doc_lines=0, code_lines=0, rust_blocks=0, text_blocks=0, tests=0)

    for name, text in sorted(mods.items()):
        api = strip_tests(text)
        fns = public_fns(api)
        n_tests = len(re.findall(r"^\s*#\[test\]", text, re.M))
        # A parameter list's arity: commas at depth 0, ignoring a leading self.
        wide = 0
        seeded = 0
        for _, params in fns:
            depth, args, cur = 0, 0, ""
            for ch in params:
                if ch in "(<[":
                    depth += 1
                elif ch in ")>]":
                    depth -= 1
                    if depth == 0:
                        break
                elif ch == "," and depth == 1:
                    args += 1
                cur += ch
            if args >= 4:  # commas at depth 1 => args+1 parameters
                wide += 1
            if re.search(r"\bseed\s*:", params):
                seeded += 1
        result = len(re.findall(r"->\s*Result<", api))
        option = len(re.findall(r"->\s*Option<", api))
        must_use = len(re.findall(r"#\[must_use", api))
        doc_lines = len(re.findall(r"^\s*(?://[/!])", api, re.M))
        code_lines = sum(1 for l in api.splitlines() if l.strip() and not l.strip().startswith("//"))
        rust_blocks = len(re.findall(r"```(?:rust|no_run|ignore|should_panic)?\s*$", api, re.M))
        text_blocks = len(re.findall(r"```text", api))
        has_mod_doc = api.lstrip().startswith("//!")

        rows.append(dict(name=name, fns=len(fns), tests=n_tests, muts=name in mutated,
                         result=result, option=option, must_use=must_use, wide=wide,
                         seeded=seeded, doc=doc_lines, code=code_lines,
                         rust=max(0, rust_blocks - text_blocks), text=text_blocks,
                         mod_doc=has_mod_doc))
        for k, v in dict(pub_fns=len(fns), result=result, option=option, must_use=must_use,
                         wide=wide, seeded=seeded, doc_lines=doc_lines, code_lines=code_lines,
                         rust_blocks=max(0, rust_blocks - text_blocks), text_blocks=text_blocks,
                         tests=n_tests).items():
            tot[k] += v

        reason = REASONS.get(name)
        if len(fns) > 0 and n_tests == 0 and not reason:
            failures.append(f"{name}: {len(fns)} public fns and NO tests, and no reason in REASONS")

        if reason and n_tests > 0 and name in mutated:
            stale.append(f"{name}: exempt in REASONS, but it now has {n_tests} tests AND a mutation row")

    n = len(rows)
    print(f"ferrotherm quality scorecard -- {n} modules, {tot['pub_fns']} public fns\n")

    print("CONSISTENCY")
    print(f"  public fns returning Result           {tot['result']:>5}")
    print(f"  public fns returning Option           {tot['option']:>5}")
    print(f"  #[must_use] annotations               {tot['must_use']:>5}")
    print(f"  public fns with 5+ parameters         {tot['wide']:>5}   (ease of use: the caller must supply all of them)")
    print(f"  public fns taking a seed              {tot['seeded']:>5}   (the caller must invent one)")
    no_mod_doc = [r['name'] for r in rows if not r['mod_doc']]
    print(f"  modules without a //! module doc      {len(no_mod_doc):>5}   {' '.join(no_mod_doc) if no_mod_doc else ''}")

    print("\nDOCUMENTATION")
    print(f"  doc lines / code lines                {tot['doc_lines'] / max(1, tot['code_lines']):>5.2f}   ({tot['doc_lines']} / {tot['code_lines']})")
    print(f"  ```text doc blocks (NOT compiled)     {tot['text_blocks']:>5}")
    print(f"  compiled doc examples                 {tot['rust_blocks']:>5}   <- these are the ones rustdoc runs")
    thin = sorted((r for r in rows if r['code'] > 200), key=lambda r: r['doc'] / max(1, r['code']))[:5]
    print("  thinnest-documented modules over 200 code lines:")
    for r in thin:
        print(f"      {r['name']:<16} {r['doc'] / max(1, r['code']):.2f} doc/code   ({r['doc']} / {r['code']})")

    print("\nTEST RIGOUR")
    print(f"  #[test] functions                     {tot['tests']:>5}")
    print(f"  modules with a recorded mutation      {len(mutated):>5} of {n}")
    bare = [r['name'] for r in rows if r['fns'] > 0 and not r['muts'] and r['name'] not in REASONS]
    print(f"  public modules with NO mutation       {len(bare):>5}")
    if bare:
        print(f"      {' '.join(sorted(bare))}")

    print("\nEASE OF USE")
    ex_dir = os.path.join(ROOT, "examples")
    exs = [f for f in os.listdir(ex_dir) if f.endswith(".rs")] if os.path.isdir(ex_dir) else []
    lens = []
    for f in exs:
        with open(os.path.join(ex_dir, f), encoding="utf-8") as fh:
            lens.append(sum(1 for l in fh if l.strip() and not l.strip().startswith("//")))
    print(f"  examples                              {len(exs):>5}")
    if lens:
        lens.sort()
        print(f"  under 50 code lines                   {sum(1 for l in lens if l < 50):>5} of {len(lens)}")
        print(f"  median example length                 {lens[len(lens) // 2]:>5} code lines")

    if stale:
        print("\nSTALE EXEMPTIONS -- these modules no longer need their entry in REASONS:")
        for s in stale:
            print(f"  {s}")
    if len(bare) > MAX_UNMUTATED:
        failures.append(
            f"mutation coverage went BACKWARDS: {len(bare)} public modules have no recorded "
            f"mutation, against a pinned {MAX_UNMUTATED}. Add a row, or lower the pin only after "
            f"the work."
        )
    elif len(bare) < MAX_UNMUTATED:
        print(
            f"\n  RATCHET: {len(bare)} unmutated modules against a pin of {MAX_UNMUTATED} -- "
            f"lower MAX_UNMUTATED to {len(bare)} in scripts/quality.py to hold the gain."
        )

    if failures:
        print(f"\nUNEXPLAINED GAPS ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        print("\nAdd the test, add the mutation, or write the reason in scripts/quality.py's REASONS.")
    else:
        print(
            f"\nNo unexplained gap: every module with public API has tests or a written reason, and "
            f"mutation coverage is at or under its pin ({len(bare)} of {MAX_UNMUTATED}).\n"
            f"That is NOT the same as complete -- {len(bare)} modules still have no recorded "
            f"mutation, which is what the pin exists to shrink."
        )

    bad = failures + stale
    return 0 if report_only or not bad else 1

if __name__ == "__main__":
    sys.exit(main())
