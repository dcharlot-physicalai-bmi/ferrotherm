"""The README's own code blocks, executed.

`python/README.md` is what PyPI serves as this package's description, and it was a **verbatim copy
of the Rust README** -- `cargo add ferrotherm` and Rust code, shown to people who ran `pip install`.
It also still carried the uncompilable `joules` snippet that the Rust README had already been fixed
for, because the fix was applied in one place and the copy was forgotten.

A README nothing executes is a README that drifts. This runs every ```python block in it, so the
quickstart is checked the way `#[cfg(doctest)]` checks the Rust one.

Writing it caught two more: `magnetization` is a property and the draft called it as a method, and
`ledger()` does not exist -- the accessors are `node_updates` and `joules`.
"""

import contextlib
import io
import re
import pathlib

import pytest

README = pathlib.Path(__file__).with_name("README.md")


def _blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", README.read_text(), re.S)


def test_the_readme_has_python_blocks_to_run():
    # A floor. If the fences are renamed or the file is replaced, the test below would silently
    # iterate over nothing and pass -- which is the shape this repository keeps finding.
    blocks = _blocks()
    assert len(blocks) >= 3, f"only {len(blocks)} python block(s); this would check almost nothing"


@pytest.mark.parametrize("i", range(len(_blocks())))
def test_readme_block_runs_and_prints_what_it_claims(i: int):
    """Each block executes standalone, and any value it claims in a comment is a value it prints.

    Running the blocks is only half of it. A block that runs while printing something other than the
    number written beside it is still a lie, and the first version of this test could not see that:
    it checked the Onsager figures against HARDCODED constants rather than against the README, so
    editing the README could not fail it. Verified by reverting the README to the 16x16/500-sweep
    parameters and watching it stay green.

    So the claims are read out of the file. A trailing `# 0.9736` on a print line is a promise about
    that line's output, and this holds the file to it.
    """
    src = _blocks()[i]
    ns: dict = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, f"README.md[python block {i + 1}]", "exec"), ns)
    out = buf.getvalue()

    # Matched PER LINE, not against the whole buffer.
    #
    # Searching all of stdout for the claimed token is not the same question, and it hid the exact
    # defect this test exists for: block 3 prints |M| then Onsager, and when |M| drifted to 0.8984
    # the claim `# 0.9736` on its line was still "found" -- in the NEXT line's output. A check that
    # any line printed the number is not a check that the right line did.
    #
    # The nth print in the source produces the nth line of output, so they pair by position.
    printed = out.splitlines()
    n = 0
    for line in src.splitlines():
        if "print(" not in line:
            continue
        here = printed[n] if n < len(printed) else ""
        n += 1
        if "#" not in line:
            continue
        claim = line.split("#", 1)[1].strip()
        for token in re.findall(r"-?\d+\.\d+", claim.split("--")[0]):
            assert token in here, (
                f"block {i + 1} claims {token} on this line:\n    {line.strip()}\n"
                f"and that line printed:\n    {here}"
            )


def test_the_claims_are_load_bearing():
    """A floor: if no block claims a number, the check above passes over nothing."""
    claimed = 0
    for src in _blocks():
        for line in src.splitlines():
            if "print(" in line and "#" in line:
                claimed += len(re.findall(r"-?\d+\.\d+", line.split("#", 1)[1].split("--")[0]))
    assert claimed >= 2, f"only {claimed} numeric claim(s) in the README; nothing is being held to"
