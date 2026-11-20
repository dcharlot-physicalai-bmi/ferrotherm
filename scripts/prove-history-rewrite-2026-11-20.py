#!/usr/bin/env python3
"""Reproduce the proof for the 2026-11-20 trailer rewrite.

Compares origin/<branch> (pre-rewrite) with rewrite/<branch> (post-rewrite).
main is not rewritten. Exits with status 1 if any check fails.
"""
import subprocess
import sys

TRAILER = b"Co-authored-by: David Charlot <dcharlot65-openie@users.noreply.github.com>"
BRANCHES = [
    "bench/thrml-tsp",
    "bench/thrml-tsp-fleet",
    "chore/toolchain-1.100",
    "fix/ci-green",
    "perf/large-grid",
]

def git(*args):
    return subprocess.check_output(["git", *args])

def parse(sha):
    raw = git("cat-file", "commit", sha)
    head, _, body = raw.partition(b"\n\n")
    tree = parents = None
    parents = []
    author = committer = None
    for line in head.splitlines():
        if line.startswith(b"tree "):
            tree = line[5:]
        elif line.startswith(b"parent "):
            parents.append(line[7:])
        elif line.startswith(b"author "):
            author = line[7:]
        elif line.startswith(b"committer "):
            committer = line[10:]
    return tree, parents, author, committer, body

def main():
    fails = {
        "tree": 0,
        "author": 0,
        "committer": 0,
        "parents": 0,
        "message": 0,
    }
    passes = {k: 0 for k in fails}
    unique_fails = {k: 0 for k in fails}
    unique_passes = {k: 0 for k in fails}
    compared = 0
    trailer_commits = 0
    message_identical = 0
    unique_trailer = 0
    unique_identical = 0
    mapping = {}
    seen = set()
    main_sha = git("rev-parse", "origin/main").decode().strip()
    local_main = git("rev-parse", "main").decode().strip()
    if local_main != main_sha:
        print(f"FAIL main moved: local {local_main} origin {main_sha}")
        return 1
    print(f"main unchanged {main_sha}")

    for branch in BRANCHES:
        old_list = git("rev-list", "--reverse", f"origin/main..origin/{branch}").decode().split()
        new_list = git("rev-list", "--reverse", f"origin/main..rewrite/{branch}").decode().split()
        if len(old_list) != len(new_list):
            print(f"FAIL {branch}: commit count {len(old_list)} vs {len(new_list)}")
            return 1
        # old tip's merge-base with main is main itself (branches are ahead only)
        mb = git("merge-base", "origin/main", f"origin/{branch}").decode().strip()
        if mb != main_sha:
            print(f"FAIL {branch}: merge-base {mb} is not main")
            return 1
        nmb = git("merge-base", "origin/main", f"rewrite/{branch}").decode().strip()
        if nmb != main_sha:
            print(f"FAIL {branch}: rewritten merge-base {nmb} is not main")
            return 1
        bodies = git("log", "--format=%B", f"rewrite/{branch}")
        trailer_n = bodies.count(b"dcharlot65-openie")
        if trailer_n != 0:
            print(f"FAIL {branch}: trailer count {trailer_n}")
            return 1
        print(f"{branch}: compared {len(old_list)} commits, trailer count on rewrite/{branch} = 0")

        for old, new in zip(old_list, new_list):
            if old in mapping and mapping[old] != new:
                print(f"FAIL parent-mapping inconsistency: {old} -> {mapping[old]} and {new}")
                fails["parents"] += 1
                return 1
            mapping[old] = new
            otree, oparents, oauthor, ocommitter, obody = parse(old)
            ntree, nparents, nauthor, ncommitter, nbody = parse(new)
            compared += 1
            first = old not in seen
            seen.add(old)
            def score(key, ok):
                if ok:
                    passes[key] += 1
                    if first:
                        unique_passes[key] += 1
                else:
                    fails[key] += 1
                    if first:
                        unique_fails[key] += 1
            # tree
            if otree == ntree:
                score("tree", True)
            else:
                score("tree", False)
                print(f"FAIL tree {old} {otree.decode()} != {ntree.decode()}")
            # author name/email/date is the whole author line
            if oauthor == nauthor:
                score("author", True)
            else:
                score("author", False)
                print(f"FAIL author {old}")
                print(f"  old {oauthor!r}")
                print(f"  new {nauthor!r}")
            if ocommitter == ncommitter:
                score("committer", True)
            else:
                score("committer", False)
                print(f"FAIL committer {old}")
                print(f"  old {ocommitter!r}")
                print(f"  new {ncommitter!r}")
            expected_parents = [mapping.get(p.decode(), p.decode()).encode() for p in oparents]
            # parents on main are absent from mapping, so they stay the original SHA
            if nparents == expected_parents:
                score("parents", True)
            else:
                score("parents", False)
                print(f"FAIL parents {old}")
                print(f"  expected {[p.decode() for p in expected_parents]}")
                print(f"  actual   {[p.decode() for p in nparents]}")
            if TRAILER in obody:
                trailer_commits += 1
                if first:
                    unique_trailer += 1
                # Message diff is exactly one blank line plus the trailer line.
                ok = obody == nbody + b"\n" + TRAILER + b"\n" and TRAILER not in nbody
                if ok:
                    score("message", True)
                else:
                    score("message", False)
                    print(f"FAIL message trailer diff {old}")
            else:
                message_identical += 1
                if first:
                    unique_identical += 1
                if obody == nbody:
                    score("message", True)
                else:
                    score("message", False)
                    print(f"FAIL message changed without a trailer {old}")
            # a rewritten commit must not still be the old object
            if old == new:
                print(f"FAIL {old} kept its SHA; an ancestor change or trailer should have moved it")
                return 1

    # every commit on main that is an ancestor of a rewritten branch is unchanged:
    # merge-base is main, so the parent of the first rewritten commit is main.
    unique = len(mapping)
    print("---")
    print(f"branch-commit comparisons: {compared}")
    print(f"unique commits: {unique}")
    print(f"trailer comparisons (message diff is the blank line + trailer): {trailer_commits}")
    print(f"unique trailer commits: {unique_trailer}")
    print(f"message-identical comparisons (parent rewrite only): {message_identical}")
    print(f"unique message-identical commits: {unique_identical}")
    print("branch-commit checks (a shared commit is checked on every branch):")
    for key in ("tree", "author", "committer", "parents", "message"):
        print(f"  {key}: pass {passes[key]} fail {fails[key]}")
    print("unique-commit checks:")
    for key in ("tree", "author", "committer", "parents", "message"):
        print(f"  {key}: pass {unique_passes[key]} fail {unique_fails[key]}")
    trailer_after = 0
    for branch in BRANCHES:
        trailer_after += git("log", "--format=%B", f"rewrite/{branch}").count(b"dcharlot65-openie")
    print(f"trailer count on new branches: {trailer_after}")
    if any(fails.values()) or trailer_after != 0 or compared == 0:
        print("PROOF FAILED")
        return 1
    print("PROOF PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
