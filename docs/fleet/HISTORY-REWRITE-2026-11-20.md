# History rewrite, 2026-11-20

Every remote branch except `main` was rewritten in one message-filter pass. The pass is the same operation as `git filter-repo --message-callback`: for each commit reachable from the branch and not from `main`, write a new commit with the same tree, the same author line, and the same committer line. The only message change is the removal of this line:

```
Co-authored-by: David Charlot <dcharlot65-openie@users.noreply.github.com>
```

and the blank line that removal leaves immediately before it. On every commit that carried the trailer, the old message was `body\n\n<trailer>\n` and the new message is `body\n`. Commits that did not carry the trailer keep their message byte-identical and get a new SHA only because a parent did. `main` (`3c474719e193d4f441201975b16419223e83d160`) was not rewritten. No remote branch other than `main` had a trailer count of 0, so none is listed as untouched.

The rewritten tips below are the branch tips immediately after the filter, before this file was added. SSH signatures on the old commits were not copied: a new message or a new parent makes the old signature not a signature of the new object. Author name, author email, author date, committer name, committer email, and committer date were copied unchanged.

`scripts/prove-history-rewrite-2026-11-20.py` reproduces the proof. Run it from a clone where `origin/<branch>` is still the pre-rewrite tip and `rewrite/<branch>` is the filtered tip, before any force-push.

## Counts

| Branch | Commits ahead of main | Commits with the trailer | Old tip | Rewritten tip |
| --- | --- | --- | --- | --- |
| `bench/thrml-tsp` | 2 | 2 | `263e62b7028af8021215ebf546ffc6f7bf1a1ad0` | `bcfd6b55ff7a6298cb066943f0c830a9234a03ad` |
| `bench/thrml-tsp-fleet` | 29 | 13 | `0f6d59ca8561cdb11c9a14d2795f1404847148a5` | `86db3fcecbe6cccac95307097d3fff43bcdf75db` |
| `chore/toolchain-1.100` | 27 | 13 | `343bd4a58ab4430734956f7413b4aaa5298392b3` | `8059b4b1605d3c5c0698025c34e8ba90a9ed389d` |
| `fix/ci-green` | 7 | 1 | `6bc29e59b4546334b68a0d3590c381de32ecf10c` | `e31d5d7af16576186088a268608db76a0c9cf47a` |
| `perf/large-grid` | 26 | 12 | `777f0417340788428cf0525003277f9829ca6e3a` | `ffe89b0702553b1c70dbbe1e52b8c537d6c0bd0b` |

45 unique commits are ahead of `main`. All 45 were rewritten. 20 of those carried the trailer. The other 25 kept a byte-identical message and moved because a parent moved. Shared commits map to one new SHA on every branch that contains them.

`main` is excluded from the rewrite. Its trailer count is 0.

## Named commits

| Role | Old SHA | New SHA |
| --- | --- | --- |
| attempt 037 publish commit | `22d3e720714c76e73389998d36e5b8159e89fe96` | `a5d0e108e46cf39bb01a2569bf114522043dc95a` |
| citation-lines commit | `df032c3adcf5679a6e12f309f7363480f7478d92` | `8b12fabfebe643bdf20a189ca7516bedd5eb7f3b` |
| bracket-proposal annotation | `4c24c5f0d443fd03bc2377f982d0b46d7f81197d` | `358f66faf279d1bd9d65e56161303fa5b240fd5f` |
| fleet tip | `0f6d59ca8561cdb11c9a14d2795f1404847148a5` | `86db3fcecbe6cccac95307097d3fff43bcdf75db` |
| side-8 pilot | `83dbb616cd9e5395f5f3d77f685fb146fc8a92d5` | `a7facbef2e2c740751313e66842abcd206385c93` |
| rules v3 freeze (`METHODOLOGY.md`) | `dbf9653a6091543df420f9ab8a3fe5da9b00a758` | `b874f6112d1eac725deca4747f41cb02e89b047e` |
| rules v3 harness, campaign-host checkout | `777f0417340788428cf0525003277f9829ca6e3a` | `ffe89b0702553b1c70dbbe1e52b8c537d6c0bd0b` |
| toolchain tip | `343bd4a58ab4430734956f7413b4aaa5298392b3` | `8059b4b1605d3c5c0698025c34e8ba90a9ed389d` |
| quiet-host harness cited as `c82c1e96` | `c82c1e96c24d6f6c292af1196cd72199b0838257` | `bb174abb3ca86e997475ecb313d625444b221c67` |

Campaign hosts run local checkouts of `777f0417340788428cf0525003277f9829ca6e3a`. This rewrite does not update those checkouts. The mapping for that SHA is in this document. Receipt lines that say the harness is `777f0417340788428cf0525003277f9829ca6e3a` still say that, so they still name the checkout the hosts have. `docs/proposals/bracket-drift-v2.md`, `CHANGELOG.md`, and `docs/toolchain-1.100.md` cite the new SHA and keep the old one as `pre_rewrite_sha`.

## Proof

```
main unchanged 3c474719e193d4f441201975b16419223e83d160
bench/thrml-tsp: compared 2 commits, trailer count on rewrite/bench/thrml-tsp = 0
bench/thrml-tsp-fleet: compared 29 commits, trailer count on rewrite/bench/thrml-tsp-fleet = 0
chore/toolchain-1.100: compared 27 commits, trailer count on rewrite/chore/toolchain-1.100 = 0
fix/ci-green: compared 7 commits, trailer count on rewrite/fix/ci-green = 0
perf/large-grid: compared 26 commits, trailer count on rewrite/perf/large-grid = 0
---
branch-commit comparisons: 91
unique commits: 45
trailer comparisons (message diff is the blank line + trailer): 41
unique trailer commits: 20
message-identical comparisons (parent rewrite only): 50
unique message-identical commits: 25
branch-commit checks (a shared commit is checked on every branch):
  tree: pass 91 fail 0
  author: pass 91 fail 0
  committer: pass 91 fail 0
  parents: pass 91 fail 0
  message: pass 91 fail 0
unique-commit checks:
  tree: pass 45 fail 0
  author: pass 45 fail 0
  committer: pass 45 fail 0
  parents: pass 45 fail 0
  message: pass 45 fail 0
trailer count on new branches: 0
PROOF PASSED
```

Tree equality is `old^{tree} == new^{tree}`. Author and committer checks compare the whole author line and the whole committer line, including the Unix timestamp and the timezone. Parent mapping: each new parent is the rewritten SHA of the old parent, or the old parent itself when that parent is on `main`. For a commit that had the trailer, the message diff is exactly the blank line plus the trailer line. For a commit that did not, the messages are byte-identical.

## Citations updated with this rewrite

On this branch (`rewrite/docs-history-2026-11-20`, parent `86db3fcecbe6cccac95307097d3fff43bcdf75db`):

- Publish commit `22d3e720714c76e73389998d36e5b8159e89fe96` now cites `a5d0e108e46cf39bb01a2569bf114522043dc95a`, with `pre_rewrite_sha`.
- Methodology commit `dbf9653a6091543df420f9ab8a3fe5da9b00a758` now cites `b874f6112d1eac725deca4747f41cb02e89b047e`, with `pre_rewrite_sha`.
- The bench README's harness citation `c82c1e96` now cites `bb174abb3ca86e997475ecb313d625444b221c67`, with `pre_rewrite_sha`. The directory name `c82c1e9-quiet24` is unchanged.
- `docs/proposals/bracket-drift-v2.md` cites the new harness SHA `ffe89b0702553b1c70dbbe1e52b8c537d6c0bd0b` and keeps `777f0417340788428cf0525003277f9829ca6e3a` as `pre_rewrite_sha`.

On `rewrite/docs-history-toolchain-2026-11-20` (parent `8059b4b1605d3c5c0698025c34e8ba90a9ed389d`): `CHANGELOG.md`, `docs/toolchain-1.100.md`, and the two commit citations in `bench/thrml-tsp/METHODOLOGY.md`.

On `rewrite/docs-history-perf-2026-11-20` (parent `ffe89b0702553b1c70dbbe1e52b8c537d6c0bd0b`): the same two `METHODOLOGY.md` citations. `rules_version` stays `v3`. No other methodology sentence changed. The sigmas `0.316246947`, `0.948740841`, `1.967373933`, `0.231989316`, `0.151394850`, `0.398473900`, and `0.806957860` are unchanged.

Left as recorded, and translated only by the tables below:

- Receipt harness lines that still name `777f0417340788428cf0525003277f9829ca6e3a`, because that is the campaign-host checkout.
- `git_rev`, `harness_sha`, and other measured-checkout fields inside result logs and JSON.
- Directory names and the historical label "the `c82c1e96` rule" inside attempt notes.
- Tarball sha256 `1df27d9f6716ca6ad0da9a0fe4c773dfc8f5a46609e5b938cc086a00d7f1b613`.

## Mapping

### `bench/thrml-tsp`

| Old SHA | New SHA | Tree | Trailer removed | Subject |
| --- | --- | --- | --- | --- |
| `e92c9fc1a263ad52862105e05175ef83e2ba7796` | `cccb7aeb95d5b3a8b9e3f4f1e03ba33e30f905f5` | `ae5e24c0873fdf42d5035fa8df1f7128f16be547` | yes | Add a THRML/TSP prior-art note and a pinned head-to-head sampler harness. |
| `263e62b7028af8021215ebf546ffc6f7bf1a1ad0` | `bcfd6b55ff7a6298cb066943f0c830a9234a03ad` | `12b871ffd55c97a61c710ff9657fc8cc996f23f3` | yes | Record the THRML head-to-head run from this VM. |

### `bench/thrml-tsp-fleet`

| Old SHA | New SHA | Tree | Trailer removed | Subject |
| --- | --- | --- | --- | --- |
| `7bb56f2fdd8b80ab7fb24bf1fbfeccb7dd59659b` | `f1ecb216f722086019f2a69e77c14e5db4726b61` | `eeb6e43665fbcfbe894f2c19cda02642694ccb4e` | yes | ci: kill the four blind mutation rows and the Euler-Mascheroni clippy error |
| `dddc4d5003f94f6d2a15436d9ca63c34866c71ab` | `5ab8f03bab821d7cac3267905c6da21e62786633` | `38be160489f522fee77528ab9c85c39d7c9117ff` | no | ci: run on fix/ci-green and bench/thrml-tsp-fleet, and on workflow_dispatch |
| `a719ab4df70c1d5cac3957e88a37496ee79ab8ad` | `1e81f75a7e0f75f7d11f9370f6ca58819cc5d434` | `9032c041b77a9f4123bd013310360a4a42b41533` | yes | Add a THRML/TSP prior-art note and a pinned head-to-head sampler harness. |
| `bfaa2f552042172e7554224e075eb21481086591` | `9d780714ef5aa7deac46b0a27c1a5afa1b641de2` | `d0b3ef7628dc92382713f5ab42e77fc315d349b5` | yes | Record the THRML head-to-head run from this VM. |
| `1c20b7ebb6a46ca401450f735d90823dab2cf73d` | `0f454d5b0c9f90584bae25ed7d9b7edef0cc608b` | `4b82ed742e4818e01f5e831f059ce2d563d78ea0` | yes | Record the fleet energy denials and keep contended joules off the quiet grade. |
| `63f752d49aafab68b43b833da0e98232cdf3cff8` | `ba4f5beb3739d04ed36a766905da8b832a482a41` | `a25c7416ec548ba80c8e9814a74e253b21011844` | yes | Publish the 189502d contended joules and refuse a short RAPL window. |
| `94c945a16bbb574fcbd4f0b9aeaace1552ce9bc4` | `81cacbd1f2bb17aee06e95a6dc3451841b60f3fe` | `79693518fb5e05e1de86d8dbcc0b37760659a9d5` | yes | Publish the 0029c27 contended joules and match POSIX read. |
| `aa857aec1f44a73acc798bada889300c9d788393` | `6dd99a127678cf0732c1bdec230c24fd0bd65adb` | `94813a2d06d25e277a116862b8a7232deff353ac` | yes | Add scaled Ising, Potts, and three-body workloads, and a CUDA energy path. |
| `111aad47076fb3b275fd949d4d35371dff385bd8` | `3eafc0769d83ac4d1558cd0761ebf0ee3e3128b2` | `8ab67a9dbff027b03bcac11d7e234ce86de0522b` | yes | Publish the invalid 8e9ad0a scale run and measure loaded hosts in pairs. |
| `f5cf8de238c020c2140c5e492296393d09e5fe37` | `681acf2b67c461bab7eaaa608978043d3f843bf1` | `f3c511a85dedf58f3f16dec468c3568d52570f7b` | no | bench: fill the initial all-down state in one call |
| `adcf448f9618886b32721052f8db345eed156f51` | `83acf1623f0b4cfa0e8313d68480cb9622e1c39f` | `fcb2fbdf4fe1a55ee3b326a2c844c5b41deeaba7` | no | Label macmon sys_power as SoC telemetry and integrate it with a trapezoid. |
| `5c54005fd0ff8dc62adedda478faa52388911902` | `2a9b0d4ab265b6a119b085a0d235cc0609f0ffec` | `2dc5d89455a07cdcb55edf87ce0933492f4fc7a3` | no | Publish the adcf448 quiet-gate refusals and measure energy before accuracy. |
| `39f04b4d7bc4e6383bc253b98ad353b9d923cc06` | `dcc7c52ed5db0cda540cd8267dad386abe9c7ee0` | `78f1964cef2e9b1068df86024fa42ed25c63c766` | no | autocorr: the GMRES rounding floor must refuse, not merely land close |
| `10ee3056116a246b019b403a402aa9861ea0334f` | `c9c4d057120520813c591d85c68e4b23b0f17fb5` | `99d61b1e860be3951d73d831f5df323d4ef9f1bb` | no | gpu: mark SoC as a name in the joules example docs |
| `6ab5c4a63c4f3736fdda327d3b4ec04ff393588a` | `c33e188521e0e2d5833e70b606716d1022105c3a` | `c9e7386fef527975f54b2c67df85fba5a8f8a62b` | no | Make the harness build and meter power on Jetson Thor. |
| `c82c1e96c24d6f6c292af1196cd72199b0838257` | `bb174abb3ca86e997475ecb313d625444b221c67` | `82596afae7959a7c425e385b087237087ed1f54d` | no | Grade energy on its own phase and hold the host lock across it. |
| `27615c0cb81f31e508fff26da2af0a36bc4d7a0e` | `4f5ac18968143c59552638f078b717ca49b2e162` | `f74a582f1de5a810a8b0d351603d429d18960a21` | no | autocorr: the GMRES floor has to show up in the refused estimate |
| `acb3b73301aa37030ed7a0092b9bffc29aaf4b1b` | `7399409e1da3d13a7c67a43fe30fb937e2641520` | `e201261086436306cf5f2c150344e94f517aefc3` | no | Publish jetson-hub attempt 3 as unguarded 5c54005 joules. |
| `068344dcda167ae75b0a3b9b15d0da258637e2a7` | `fdeb222f81ba00d412a53c2bb28ae62e3ebddd79` | `23a4b722380804ec6d77c1ed9009840ba23ed622` | no | Bind the small-model joule quotients to the sizes they cover. |
| `2838c3e172ccfd68f5b97b6d08e163a3780d82d1` | `80873c60407f8e85cdde577e4a4f656485110e67` | `78b68f3f83c1a435c41e2a915876bf121a6504f3` | no | Follow hwmon symlinks when discovering Jetson INA238 VIN. |
| `01be0a3b3f8546922ad1503b883434148f179268` | `82069aa8b04055fa4a55df6dd88d1e0369e9e025` | `69371ed122fc7a9945d286ae40f61ed972f79bc7` | no | Label AMD RAPL as modeled SoC energy and stamp the macmon version. |
| `fe36c71e062494c3949da20bea4733e910184a06` | `172715a48efa937ae2b54622cf95ce519b7b109e` | `348b4e69e3fecd01f91fa97d3a9dfd136ba5c0bd` | no | autocorr: compare the GMRES estimate with the floor, not with a host constant |
| `4d674ffb36f048a86c16c86095646a605a7b3a43` | `b49e14278b3eb4600896abdae9ae8fb2231a2aeb` | `a06fee562cd85a9258c276dc2ec58c67e47f3a3d` | no | autocorr: pass Cell::get to the GMRES estimate cell |
| `bf14c1bcffdc09fea537f5641a872d2bc1e49bb1` | `b93a6d3679febc2a26daf23184b2bfd590cef0d5` | `c762ee5b97106d11c25969979cf7b478a05102e7` | no | autocorr: the GMRES floor only has to beat the raw residual |
| `22d3e720714c76e73389998d36e5b8159e89fe96` | `a5d0e108e46cf39bb01a2569bf114522043dc95a` | `08c4fe9264e2f8dbb87f892b735eb534d4cf3157` | yes | Publish jetson-hub attempt 037 as quiet-host Intel RAPL core joules. |
| `df032c3adcf5679a6e12f309f7363480f7478d92` | `8b12fabfebe643bdf20a189ca7516bedd5eb7f3b` | `e97fcd575ecaa54b96eb8803d033a206a53faef6` | yes | Name the attempt 037 publish commit in the citation lines. |
| `ac1ef69299a203cf253d41d435d627b490c310dd` | `b97a0280dc0dd87fdbf16843b929b3b08efa5220` | `c91e8043ee3b968708d1ef25d025eefbe321966f` | yes | Audit scoring inside the attempt 037 energy window. |
| `4c24c5f0d443fd03bc2377f982d0b46d7f81197d` | `358f66faf279d1bd9d65e56161303fa5b240fd5f` | `640e216aaf9af37bb76129a48547ab702db57021` | yes | Annotate attempt 037 as measured and supersede the bracket proposal. |
| `0f6d59ca8561cdb11c9a14d2795f1404847148a5` | `86db3fcecbe6cccac95307097d3fff43bcdf75db` | `15ca4498dfb2be1c62b515f9cb406d9e90f7e2f9` | yes | Cite the frozen rules v3 commits in the attempt 037 notes. |

### `chore/toolchain-1.100`

| Old SHA | New SHA | Tree | Trailer removed | Subject |
| --- | --- | --- | --- | --- |
| `7bb56f2fdd8b80ab7fb24bf1fbfeccb7dd59659b` | `f1ecb216f722086019f2a69e77c14e5db4726b61` | `eeb6e43665fbcfbe894f2c19cda02642694ccb4e` | yes | ci: kill the four blind mutation rows and the Euler-Mascheroni clippy error |
| `dddc4d5003f94f6d2a15436d9ca63c34866c71ab` | `5ab8f03bab821d7cac3267905c6da21e62786633` | `38be160489f522fee77528ab9c85c39d7c9117ff` | no | ci: run on fix/ci-green and bench/thrml-tsp-fleet, and on workflow_dispatch |
| `a719ab4df70c1d5cac3957e88a37496ee79ab8ad` | `1e81f75a7e0f75f7d11f9370f6ca58819cc5d434` | `9032c041b77a9f4123bd013310360a4a42b41533` | yes | Add a THRML/TSP prior-art note and a pinned head-to-head sampler harness. |
| `bfaa2f552042172e7554224e075eb21481086591` | `9d780714ef5aa7deac46b0a27c1a5afa1b641de2` | `d0b3ef7628dc92382713f5ab42e77fc315d349b5` | yes | Record the THRML head-to-head run from this VM. |
| `1c20b7ebb6a46ca401450f735d90823dab2cf73d` | `0f454d5b0c9f90584bae25ed7d9b7edef0cc608b` | `4b82ed742e4818e01f5e831f059ce2d563d78ea0` | yes | Record the fleet energy denials and keep contended joules off the quiet grade. |
| `63f752d49aafab68b43b833da0e98232cdf3cff8` | `ba4f5beb3739d04ed36a766905da8b832a482a41` | `a25c7416ec548ba80c8e9814a74e253b21011844` | yes | Publish the 189502d contended joules and refuse a short RAPL window. |
| `94c945a16bbb574fcbd4f0b9aeaace1552ce9bc4` | `81cacbd1f2bb17aee06e95a6dc3451841b60f3fe` | `79693518fb5e05e1de86d8dbcc0b37760659a9d5` | yes | Publish the 0029c27 contended joules and match POSIX read. |
| `aa857aec1f44a73acc798bada889300c9d788393` | `6dd99a127678cf0732c1bdec230c24fd0bd65adb` | `94813a2d06d25e277a116862b8a7232deff353ac` | yes | Add scaled Ising, Potts, and three-body workloads, and a CUDA energy path. |
| `111aad47076fb3b275fd949d4d35371dff385bd8` | `3eafc0769d83ac4d1558cd0761ebf0ee3e3128b2` | `8ab67a9dbff027b03bcac11d7e234ce86de0522b` | yes | Publish the invalid 8e9ad0a scale run and measure loaded hosts in pairs. |
| `f5cf8de238c020c2140c5e492296393d09e5fe37` | `681acf2b67c461bab7eaaa608978043d3f843bf1` | `f3c511a85dedf58f3f16dec468c3568d52570f7b` | no | bench: fill the initial all-down state in one call |
| `adcf448f9618886b32721052f8db345eed156f51` | `83acf1623f0b4cfa0e8313d68480cb9622e1c39f` | `fcb2fbdf4fe1a55ee3b326a2c844c5b41deeaba7` | no | Label macmon sys_power as SoC telemetry and integrate it with a trapezoid. |
| `5c54005fd0ff8dc62adedda478faa52388911902` | `2a9b0d4ab265b6a119b085a0d235cc0609f0ffec` | `2dc5d89455a07cdcb55edf87ce0933492f4fc7a3` | no | Publish the adcf448 quiet-gate refusals and measure energy before accuracy. |
| `39f04b4d7bc4e6383bc253b98ad353b9d923cc06` | `dcc7c52ed5db0cda540cd8267dad386abe9c7ee0` | `78f1964cef2e9b1068df86024fa42ed25c63c766` | no | autocorr: the GMRES rounding floor must refuse, not merely land close |
| `10ee3056116a246b019b403a402aa9861ea0334f` | `c9c4d057120520813c591d85c68e4b23b0f17fb5` | `99d61b1e860be3951d73d831f5df323d4ef9f1bb` | no | gpu: mark SoC as a name in the joules example docs |
| `6ab5c4a63c4f3736fdda327d3b4ec04ff393588a` | `c33e188521e0e2d5833e70b606716d1022105c3a` | `c9e7386fef527975f54b2c67df85fba5a8f8a62b` | no | Make the harness build and meter power on Jetson Thor. |
| `c82c1e96c24d6f6c292af1196cd72199b0838257` | `bb174abb3ca86e997475ecb313d625444b221c67` | `82596afae7959a7c425e385b087237087ed1f54d` | no | Grade energy on its own phase and hold the host lock across it. |
| `27615c0cb81f31e508fff26da2af0a36bc4d7a0e` | `4f5ac18968143c59552638f078b717ca49b2e162` | `f74a582f1de5a810a8b0d351603d429d18960a21` | no | autocorr: the GMRES floor has to show up in the refused estimate |
| `acb3b73301aa37030ed7a0092b9bffc29aaf4b1b` | `7399409e1da3d13a7c67a43fe30fb937e2641520` | `e201261086436306cf5f2c150344e94f517aefc3` | no | Publish jetson-hub attempt 3 as unguarded 5c54005 joules. |
| `8b709d28b1eb82b1cc1de688c51fc28adcfb6efc` | `e572b6a447a9df15977c6ac053b4d1690dabbe05` | `81f1d96f3a2b8773833ea7d9260c01d3a017337a` | no | Add open-grid sampler levers and a second THRML baseline. |
| `9c6d1ef073641ea12f1923c671a7186b95d8a045` | `44d365665552edf475db46db0e79b26f45c63525` | `e0d5376327080d08accfad89c2e04ec9b3f25234` | no | Require --cpus on grid-scale energy runs and record the VM limit. |
| `d807fbe12e664c823883d852dcfaa38ed9129fe0` | `f56aec67d48a83d19563cc64f175c76659cd1eac` | `333321669e6ccd36af0a0d16d26b26bc2be66859` | no | Combine the open-grid field levers and record a THRML OOM as its own limit. |
| `16bf3988029c5bf5ffd048b4f2715bbc7209afd0` | `35f4beb6d5442df06e630d15432c939581cd7b87` | `9e1998d562f3ce100ab33d137208078dff29c457` | no | Record THRML ns/sweep, label CUDA OOM, and batch the packed GPU chain. |
| `4a3c2607a78d8910525712094e5adb41bc66337c` | `a144d3e69ec2771cc43bfb4751bd59202c2ee176` | `bd5afe1995297cf162b9b3decb7ceb53961f093b` | yes | Give side 8 a powered two-seed sample and hash every receipt. |
| `83dbb616cd9e5395f5f3d77f685fb146fc8a92d5` | `a7facbef2e2c740751313e66842abcd206385c93` | `3bb6eb0328f3d7b6368a33ffe942bf7f8f6b505b` | yes | Record the side-8 pilot in the receipt and add quiet grid-energy. |
| `dbf9653a6091543df420f9ab8a3fe5da9b00a758` | `b874f6112d1eac725deca4747f41cb02e89b047e` | `7538e3ea3a2888f3642ad07a0d21545542593e47` | yes | Freeze measurement rules v3 before any campaign uses them. |
| `777f0417340788428cf0525003277f9829ca6e3a` | `ffe89b0702553b1c70dbbe1e52b8c537d6c0bd0b` | `90b752e1d2a922aa304aab468e2f9328014e98fc` | yes | Apply measurement rules v3 to both energy suites. |
| `343bd4a58ab4430734956f7413b4aaa5298392b3` | `8059b4b1605d3c5c0698025c34e8ba90a9ed389d` | `d70cc92e2bfa61fceaeb0630afeff77e88afcbbc` | yes | Pin Rust 1.100.0 and current dependency releases. |

### `fix/ci-green`

| Old SHA | New SHA | Tree | Trailer removed | Subject |
| --- | --- | --- | --- | --- |
| `7bb56f2fdd8b80ab7fb24bf1fbfeccb7dd59659b` | `f1ecb216f722086019f2a69e77c14e5db4726b61` | `eeb6e43665fbcfbe894f2c19cda02642694ccb4e` | yes | ci: kill the four blind mutation rows and the Euler-Mascheroni clippy error |
| `dddc4d5003f94f6d2a15436d9ca63c34866c71ab` | `5ab8f03bab821d7cac3267905c6da21e62786633` | `38be160489f522fee77528ab9c85c39d7c9117ff` | no | ci: run on fix/ci-green and bench/thrml-tsp-fleet, and on workflow_dispatch |
| `b2591969dadbcaf1980e8535d4b358c2bcee42cc` | `197120d2c6233b95ccb0e9936aca09a23a1e1a84` | `ecb7acd672c4437fa5efd8703b89420c36bfc4b7` | no | autocorr: the GMRES rounding floor must refuse, not merely land close |
| `f4edbf1ce57b7b834049a5025501edb2073248a1` | `5fd63b2478ecabd866cfd51918b7ab6e7ff79561` | `d2bfa945d29b4e3151e7e251181e0ebbc5e3d1e0` | no | autocorr: the GMRES floor has to show up in the refused estimate |
| `3f34b849a5802917c49886e7be768b0427e92359` | `8d28b747917cf0f0ac8f8cee2178a6d984f89b4a` | `edded36982a46163d7a665fea95a53759b50580c` | no | autocorr: compare the GMRES estimate with the floor, not with a host constant |
| `65dd0a077161b95ebf60d305a28a95d607ab053b` | `730fc6b45b10883ebfb10d1de847dd9528656134` | `ec1865b3f7788b1c5743b3f74f984f2b15f21672` | no | autocorr: pass Cell::get to the GMRES estimate cell |
| `6bc29e59b4546334b68a0d3590c381de32ecf10c` | `e31d5d7af16576186088a268608db76a0c9cf47a` | `7a1dff199e8e589a358f160c186f9b172f8d0593` | no | autocorr: the GMRES floor only has to beat the raw residual |

### `perf/large-grid`

| Old SHA | New SHA | Tree | Trailer removed | Subject |
| --- | --- | --- | --- | --- |
| `7bb56f2fdd8b80ab7fb24bf1fbfeccb7dd59659b` | `f1ecb216f722086019f2a69e77c14e5db4726b61` | `eeb6e43665fbcfbe894f2c19cda02642694ccb4e` | yes | ci: kill the four blind mutation rows and the Euler-Mascheroni clippy error |
| `dddc4d5003f94f6d2a15436d9ca63c34866c71ab` | `5ab8f03bab821d7cac3267905c6da21e62786633` | `38be160489f522fee77528ab9c85c39d7c9117ff` | no | ci: run on fix/ci-green and bench/thrml-tsp-fleet, and on workflow_dispatch |
| `a719ab4df70c1d5cac3957e88a37496ee79ab8ad` | `1e81f75a7e0f75f7d11f9370f6ca58819cc5d434` | `9032c041b77a9f4123bd013310360a4a42b41533` | yes | Add a THRML/TSP prior-art note and a pinned head-to-head sampler harness. |
| `bfaa2f552042172e7554224e075eb21481086591` | `9d780714ef5aa7deac46b0a27c1a5afa1b641de2` | `d0b3ef7628dc92382713f5ab42e77fc315d349b5` | yes | Record the THRML head-to-head run from this VM. |
| `1c20b7ebb6a46ca401450f735d90823dab2cf73d` | `0f454d5b0c9f90584bae25ed7d9b7edef0cc608b` | `4b82ed742e4818e01f5e831f059ce2d563d78ea0` | yes | Record the fleet energy denials and keep contended joules off the quiet grade. |
| `63f752d49aafab68b43b833da0e98232cdf3cff8` | `ba4f5beb3739d04ed36a766905da8b832a482a41` | `a25c7416ec548ba80c8e9814a74e253b21011844` | yes | Publish the 189502d contended joules and refuse a short RAPL window. |
| `94c945a16bbb574fcbd4f0b9aeaace1552ce9bc4` | `81cacbd1f2bb17aee06e95a6dc3451841b60f3fe` | `79693518fb5e05e1de86d8dbcc0b37760659a9d5` | yes | Publish the 0029c27 contended joules and match POSIX read. |
| `aa857aec1f44a73acc798bada889300c9d788393` | `6dd99a127678cf0732c1bdec230c24fd0bd65adb` | `94813a2d06d25e277a116862b8a7232deff353ac` | yes | Add scaled Ising, Potts, and three-body workloads, and a CUDA energy path. |
| `111aad47076fb3b275fd949d4d35371dff385bd8` | `3eafc0769d83ac4d1558cd0761ebf0ee3e3128b2` | `8ab67a9dbff027b03bcac11d7e234ce86de0522b` | yes | Publish the invalid 8e9ad0a scale run and measure loaded hosts in pairs. |
| `f5cf8de238c020c2140c5e492296393d09e5fe37` | `681acf2b67c461bab7eaaa608978043d3f843bf1` | `f3c511a85dedf58f3f16dec468c3568d52570f7b` | no | bench: fill the initial all-down state in one call |
| `adcf448f9618886b32721052f8db345eed156f51` | `83acf1623f0b4cfa0e8313d68480cb9622e1c39f` | `fcb2fbdf4fe1a55ee3b326a2c844c5b41deeaba7` | no | Label macmon sys_power as SoC telemetry and integrate it with a trapezoid. |
| `5c54005fd0ff8dc62adedda478faa52388911902` | `2a9b0d4ab265b6a119b085a0d235cc0609f0ffec` | `2dc5d89455a07cdcb55edf87ce0933492f4fc7a3` | no | Publish the adcf448 quiet-gate refusals and measure energy before accuracy. |
| `39f04b4d7bc4e6383bc253b98ad353b9d923cc06` | `dcc7c52ed5db0cda540cd8267dad386abe9c7ee0` | `78f1964cef2e9b1068df86024fa42ed25c63c766` | no | autocorr: the GMRES rounding floor must refuse, not merely land close |
| `10ee3056116a246b019b403a402aa9861ea0334f` | `c9c4d057120520813c591d85c68e4b23b0f17fb5` | `99d61b1e860be3951d73d831f5df323d4ef9f1bb` | no | gpu: mark SoC as a name in the joules example docs |
| `6ab5c4a63c4f3736fdda327d3b4ec04ff393588a` | `c33e188521e0e2d5833e70b606716d1022105c3a` | `c9e7386fef527975f54b2c67df85fba5a8f8a62b` | no | Make the harness build and meter power on Jetson Thor. |
| `c82c1e96c24d6f6c292af1196cd72199b0838257` | `bb174abb3ca86e997475ecb313d625444b221c67` | `82596afae7959a7c425e385b087237087ed1f54d` | no | Grade energy on its own phase and hold the host lock across it. |
| `27615c0cb81f31e508fff26da2af0a36bc4d7a0e` | `4f5ac18968143c59552638f078b717ca49b2e162` | `f74a582f1de5a810a8b0d351603d429d18960a21` | no | autocorr: the GMRES floor has to show up in the refused estimate |
| `acb3b73301aa37030ed7a0092b9bffc29aaf4b1b` | `7399409e1da3d13a7c67a43fe30fb937e2641520` | `e201261086436306cf5f2c150344e94f517aefc3` | no | Publish jetson-hub attempt 3 as unguarded 5c54005 joules. |
| `8b709d28b1eb82b1cc1de688c51fc28adcfb6efc` | `e572b6a447a9df15977c6ac053b4d1690dabbe05` | `81f1d96f3a2b8773833ea7d9260c01d3a017337a` | no | Add open-grid sampler levers and a second THRML baseline. |
| `9c6d1ef073641ea12f1923c671a7186b95d8a045` | `44d365665552edf475db46db0e79b26f45c63525` | `e0d5376327080d08accfad89c2e04ec9b3f25234` | no | Require --cpus on grid-scale energy runs and record the VM limit. |
| `d807fbe12e664c823883d852dcfaa38ed9129fe0` | `f56aec67d48a83d19563cc64f175c76659cd1eac` | `333321669e6ccd36af0a0d16d26b26bc2be66859` | no | Combine the open-grid field levers and record a THRML OOM as its own limit. |
| `16bf3988029c5bf5ffd048b4f2715bbc7209afd0` | `35f4beb6d5442df06e630d15432c939581cd7b87` | `9e1998d562f3ce100ab33d137208078dff29c457` | no | Record THRML ns/sweep, label CUDA OOM, and batch the packed GPU chain. |
| `4a3c2607a78d8910525712094e5adb41bc66337c` | `a144d3e69ec2771cc43bfb4751bd59202c2ee176` | `bd5afe1995297cf162b9b3decb7ceb53961f093b` | yes | Give side 8 a powered two-seed sample and hash every receipt. |
| `83dbb616cd9e5395f5f3d77f685fb146fc8a92d5` | `a7facbef2e2c740751313e66842abcd206385c93` | `3bb6eb0328f3d7b6368a33ffe942bf7f8f6b505b` | yes | Record the side-8 pilot in the receipt and add quiet grid-energy. |
| `dbf9653a6091543df420f9ab8a3fe5da9b00a758` | `b874f6112d1eac725deca4747f41cb02e89b047e` | `7538e3ea3a2888f3642ad07a0d21545542593e47` | yes | Freeze measurement rules v3 before any campaign uses them. |
| `777f0417340788428cf0525003277f9829ca6e3a` | `ffe89b0702553b1c70dbbe1e52b8c537d6c0bd0b` | `90b752e1d2a922aa304aab468e2f9328014e98fc` | yes | Apply measurement rules v3 to both energy suites. |

## Force-with-lease commands

These commands were not run. They replace each existing branch with the filtered tip above, and they refuse to do so if the remote branch is no longer at the old tip. They do not update `main`. The docs branches are not in this list.

```
git push --force-with-lease=bench/thrml-tsp:263e62b7028af8021215ebf546ffc6f7bf1a1ad0 origin rewrite/bench/thrml-tsp:bench/thrml-tsp
git push --force-with-lease=bench/thrml-tsp-fleet:0f6d59ca8561cdb11c9a14d2795f1404847148a5 origin rewrite/bench/thrml-tsp-fleet:bench/thrml-tsp-fleet
git push --force-with-lease=chore/toolchain-1.100:343bd4a58ab4430734956f7413b4aaa5298392b3 origin rewrite/chore/toolchain-1.100:chore/toolchain-1.100
git push --force-with-lease=fix/ci-green:6bc29e59b4546334b68a0d3590c381de32ecf10c origin rewrite/fix/ci-green:fix/ci-green
git push --force-with-lease=perf/large-grid:777f0417340788428cf0525003277f9829ca6e3a origin rewrite/perf/large-grid:perf/large-grid
```
