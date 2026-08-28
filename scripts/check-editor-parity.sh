#!/usr/bin/env bash
#
# Can the visual editor say everything the model layer can?
#
# scripts/check-parity.sh asks whether every C ABI symbol reaches every LANGUAGE binding. It does
# not ask about the node editor, and the node editor is a surface like any other -- so this gap sat
# open and invisible: the model layer had nine constraints, the C ABI reached all nine through
# ft_model_close's kind codes, and the editor offered six. Nothing compared the two lists, because
# a missing node type is not a build error anywhere. It is simply a thing a modeller cannot say.
#
# So the rule check-parity.sh applies to bindings is applied here to the editor, in the same shape:
# every Constraint variant in src/model.rs is reachable from a node, every Encoding is offered in
# the picker, every constraint can be priced -- and a genuine exception goes in EXEMPT WITH THE
# REASON, which is the part that matters.
#
#   scripts/check-editor-parity.sh

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

MODEL=src/model.rs
# EDITOR is not assigned here: --selftest below re-runs this script with it pointed at a damaged
# copy, and an unconditional assignment at the top would quietly overwrite that -- which is how the
# first version of this self-test reported the gate passing an editor with a node type cut out.

# --selftest runs the gate against an editor with one node type cut out of it, and expects it to
# FAIL. A gate that has only ever passed is indistinguishable from a gate that cannot fail, and
# this one is a pile of greps over a moving file -- exactly the shape that silently stops matching.
if [ "${1:-}" = "--selftest" ]; then
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  # Take out all_different the way a regression would: the node stops being declared and offered.
  sed -e '/data-t="alldifferent"/d' \
      -e 's/^  alldifferent:{ title: "All different"/  DELETED_alldifferent:{ title: "gone"/' \
      docs/graph.html > "$tmp/graph.html"
  if EDITOR="$tmp/graph.html" "$0" >/dev/null 2>&1; then
    echo "SELFTEST FAILED: the gate passed an editor missing all_different"
    exit 1
  fi
  echo "selftest ok: removing a node type makes the gate fail"
  exit 0
fi
EDITOR="${EDITOR:-docs/graph.html}"

# what|why it is not in the editor
EXEMPT=$(cat <<'TABLE'
Encoding::Binary|The compiler refuses a binary-encoded variable in any constraint or objective: a binary indicator is a product of every bit, so its degree grows with the domain. A binary variable in the editor could be declared and never used, so the picker would be offering a guaranteed refusal rather than a capability.
TABLE
)
exempt_why() { printf '%s\n' "$EXEMPT" | awk -F'|' -v k="$1" '$1 == k { print $2 }'; }

# Constraint variant : the node type that states it. The editor names things in its own vocabulary,
# so the mapping is written down once, here, and every variant must appear on the left.
MAP="NotEqual:notequal Equal:equal Fix:fix Cardinality:cardinality AtMost:atmost AtLeast:atleast
     ExactlyOne:exactlyone AtMostOne:atmostone AllDifferent:alldifferent"

fail=0
note() { printf '  %s\n' "$1"; }

echo "constraints"
for v in $(awk '/^pub enum Constraint/,/^}/' "$MODEL" | grep -oE '^    [A-Z][A-Za-z]+' | tr -d ' '); do
  node=$(printf '%s\n' $MAP | awk -F: -v v="$v" '$1 == v { print $2 }')
  if [ -z "$node" ]; then
    why=$(exempt_why "Constraint::$v")
    if [ -n "$why" ]; then note "exempt  $v -- $why"; else
      note "MISSING $v maps to no node type, and has no reason in EXEMPT"; fail=1; fi
    continue
  fi
  declared=$(grep -cE "^  ${node}:? *\{|^  ${node}: +\{" "$EDITOR" || true)
  palette=$(grep -c "data-t=\"${node}\"" "$EDITOR" || true)
  # Pairwise nodes are dispatched by name; counting nodes through the COUNTING table, which the
  # compile step reads with `n.type in COUNTING`.
  wired=$(grep -cE "n\.type === \"${node}\"|(^| )${node}: [0-9]" "$EDITOR" || true)
  if [ "$declared" -ge 1 ] && [ "$palette" -ge 1 ] && [ "$wired" -ge 1 ]; then
    note "ok      $v -> $node"
  else
    note "BROKEN  $v -> $node (declared=$declared palette=$palette wired=$wired)"; fail=1
  fi
done

echo "encodings"
picker=$(grep -o 'pick:[a-z|-]*' "$EDITOR" | head -1 | cut -d: -f2)
[ -n "$picker" ] || { note "MISSING no encoding picker in the editor at all"; fail=1; }
for e in $(awk '/pub enum Encoding/,/^}/' src/encode.rs | grep -oE '^    [A-Z][A-Za-z]+' | tr -d ' '); do
  slug=$(printf '%s\n' "$e" | sed -E 's/([a-z0-9])([A-Z])/\1-\2/g' | tr '[:upper:]' '[:lower:]')
  if printf '%s\n' "$picker" | tr '|' '\n' | grep -qx "$slug"; then
    note "ok      $e -> $slug"
  else
    why=$(exempt_why "Encoding::$e")
    if [ -n "$why" ]; then note "exempt  $e -- $why"; else
      note "MISSING $e is not offered in the picker, and has no reason in EXEMPT"; fail=1; fi
  fi
done

# A rule and a priced preference are different models. The editor could state only the first.
echo "soft"
unpriced=""
for node in $(printf '%s\n' $MAP | cut -d: -f2); do
  awk -v n="$node" '
    $0 ~ "^  " n ":" { on = 1 }
    on { buf = buf $0 }
    on && /\},[ ]*$/ { print buf; exit }
  ' "$EDITOR" | grep -q '\["soft"' || unpriced="$unpriced $node"
done
if [ -n "$unpriced" ]; then
  note "MISSING these constraints cannot be priced as preferences:$unpriced"; fail=1
else
  note "ok      every constraint node carries a soft price"
fi

echo
if [ "$fail" = 0 ]; then echo "EDITOR PARITY ok"; else echo "EDITOR PARITY failed"; fi
exit $fail
