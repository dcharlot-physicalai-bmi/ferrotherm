#!/bin/sh
# Install THRML main @ 2d96f4b4 into its own venv. Does not touch the 0.1.4 venv.
# Usage: install-baseline-b.sh VENV_DIR [jax extra, default cpu]
set -eu
dest="${1:?venv directory}"
extra="${2:-cpu}"
commit="2d96f4b4b67bfa2625d89ed79f0a22c08af3b992"
python3 -m venv "$dest"
"$dest/bin/pip" install --upgrade pip
"$dest/bin/pip" install "jax[${extra}]==0.12.0.1" "jaxlib==0.12.0.1" "equinox==0.13.8"
"$dest/bin/pip" install "thrml @ git+https://github.com/extropic-ai/thrml.git@${commit}"
"$dest/bin/python" - <<PY
import json, importlib.metadata
text = importlib.metadata.distribution("thrml").read_text("direct_url.json")
commit = json.loads(text)["vcs_info"]["commit_id"]
assert commit == "${commit}", commit
from thrml.factor import FactorSamplingProgram
from thrml.models.ising import IsingSamplingProgram
assert not issubclass(IsingSamplingProgram, FactorSamplingProgram)
print("baseline B", commit)
PY
