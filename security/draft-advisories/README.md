# Draft RustSec advisories — NOT SUBMITTED

These are drafts. **Nothing here has been filed**, and no advisory ID has been assigned; the
`RUSTSEC-0000-0000` in each is the placeholder the template uses until a maintainer assigns one on
merge.

Filing means opening a pull request against [`rustsec/advisory-db`](https://github.com/rustsec/advisory-db),
copying the file to `crates/<crate-name>/`. That publishes into the ecosystem database under your
identity, which is why it is a decision rather than a step, and why these sit here for review first.

Both describe defects that are **already fixed and released**. The question they answer is whether
someone who pinned an older version finds out, since `cargo audit` is the only channel that reaches
them.

| draft | crate | affected | fixed in |
|---|---|---|---|
| `ferrotherm-parser-panics.md` | `ferrotherm` | `<= 0.13.0` | 0.14.0 |
| `ferrotherm-serve-dos.md` | `ferrotherm-serve` | `<= 0.7.1` | 0.7.2 |

Yanking is the other half and is independent: it stops **new** dependency resolution from selecting
an affected version, while an advisory tells people who already depend on one. Neither replaces the
other.
