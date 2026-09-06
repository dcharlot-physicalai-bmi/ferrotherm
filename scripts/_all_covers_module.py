#!/usr/bin/env python3
"""Every public name this module defines must be in ``__all__``.

The hole this closes: ``scripts/gen-stubs.py`` builds the type stub from ``ferrotherm.__all__``, so
a public name that never got added to that list is invisible to the generator -- and
``check-stubs.sh`` then compares two files that agree about a library neither describes, and prints
"the stub matches". That is exactly what happened here. ``Prices``, ``Cost`` and ``PRICES`` shipped
as public API, the stub did not move a line, the gate stayed green, and every type checker and
editor completed as though they did not exist. The same silence had been hiding ``from_ommx``,
which Julia has exported all along.

A name is this module's own if it is a class or function declared here, or an UPPER_CASE constant
that knows no other home. Imported modules and re-exported third-party objects are somebody else's
surface, and claiming them would put their names in our stub.

  scripts/_all_covers_module.py              check the real module
  scripts/_all_covers_module.py --selftest   prove the rule can say no
"""

import inspect
import io
import sys


def missing_from_all(namespace, declared):
    """Public names in ``namespace`` that this module defines and ``declared`` does not list."""
    out = []
    for name, obj in sorted(namespace.items()):
        if name.startswith("_") or inspect.ismodule(obj):
            continue
        home = getattr(obj, "__module__", None)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            own = home == "ferrotherm"
        else:
            # A module-level constant has no __module__ to consult, so the naming convention stands
            # in -- but `ctypes.POINTER` is an UPPER_CASE builtin this module imports, and it
            # reports `_ctypes` as its home. Anything that knows where it came from, and did not
            # come from here, is theirs.
            own = name.isupper() and home is None
        if own and name not in declared:
            out.append(name)
    return out


def unannotated_slots(namespace, declared):
    """Exported classes whose ``__slots__`` carry attributes no annotation describes.

    The second half of the same hole. ``gen-stubs.py`` now emits class-level annotations, which is
    how a ``__slots__`` class declares its attributes — but ``__slots__`` alone carries no types, so
    a slot added without an annotation is missing from the stub and from the freshly generated one
    alike, and the byte comparison agrees about a class neither describes. That is exactly how
    ``Answer`` came to be documented by an ``__init__(**kw)`` and one property, with nothing saying
    ``answer.energy`` exists.
    """
    out = []
    for name in sorted(declared):
        obj = namespace.get(name)
        if not inspect.isclass(obj):
            continue
        slots = getattr(obj, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        ann = getattr(obj, "__annotations__", {})
        for s in slots:
            if not s.startswith("_") and s not in ann:
                out.append(f"{name}.{s}")
    return out


def undescribed_attributes(namespace, declared, source):
    """Public instance attributes an exported class sets but never annotates.

    The slot rule above only looks at ``__slots__``, so a class that does not use them was exempt
    entirely -- ``Hubo``, ``Model``, ``Problem``, ``Sim`` and ``SampleSet`` are all in ``__all__``,
    all set public attributes in ``__init__``, and all passed. ``gen-stubs.py`` emits class-level
    lines only from ``__annotations__``, so those attributes are missing from the stub and from the
    freshly generated one alike, and the byte comparison agrees about a class neither describes.

    Found by reading each class's ``__init__`` in the module source: every ``self.NAME = ...`` whose
    NAME is public and has no annotation on the class.
    """
    import ast

    tree = ast.parse(source)
    out = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in declared:
            continue
        obj = namespace.get(node.name)
        if not inspect.isclass(obj):
            continue
        ann = set(getattr(obj, "__annotations__", {}))
        slots = getattr(obj, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        # A slotted class is the other rule's subject; do not report it twice.
        if slots:
            continue
        assigned = set()
        for fn in node.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name != "__init__":
                continue
            for sub in ast.walk(fn):
                if not isinstance(sub, ast.Assign):
                    continue
                for tgt in sub.targets:
                    if (
                        isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                        and not tgt.attr.startswith("_")
                    ):
                        assigned.add(tgt.attr)
        for a in sorted(assigned - ann):
            out.append(f"{node.name}.{a}")
    return out


def _selftest():
    """A rule that cannot fail is the same evidence as no rule."""
    class Kept:
        pass
    Kept.__module__ = "ferrotherm"

    class Forgotten:
        pass
    Forgotten.__module__ = "ferrotherm"

    ns = {"Kept": Kept, "Forgotten": Forgotten, "TABLE": {"a": 1}, "_private": Kept,
          "POINTER": inspect.isclass}
    got = missing_from_all(ns, {"Kept"})
    assert got == ["Forgotten", "TABLE"], f"the rule reported {got}"
    # And a name that is listed, or private, or foreign, must NOT be reported.
    assert missing_from_all(ns, {"Kept", "Forgotten", "TABLE"}) == []
    print("selftest: a public name left out of __all__ is caught, and an imported one is not")

    class Slotted:
        __slots__ = ("described", "bare", "_private")
        described: int
    Slotted.__module__ = "ferrotherm"

    got2 = unannotated_slots({"Slotted": Slotted}, {"Slotted"})
    assert got2 == ["Slotted.bare"], f"the slot rule reported {got2}"
    print("selftest: a slot with no annotation is caught, and a private one is not")

    # And the rule for classes that use no __slots__ at all, which the one above cannot see.
    class Plain:
        described: int

        def __init__(self):
            self.described = 1
            self.bare = 2
            self._private = 3
    Plain.__module__ = "ferrotherm"

    src = (
        "class Plain:\n"
        "    described: int\n"
        "    def __init__(self):\n"
        "        self.described = 1\n"
        "        self.bare = 2\n"
        "        self._private = 3\n"
    )
    got3 = undescribed_attributes({"Plain": Plain}, {"Plain"}, src)
    assert got3 == ["Plain.bare"], f"the attribute rule reported {got3}"
    print("selftest: a public attribute on an unslotted class is caught too")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)

    sys.path.insert(0, "python")
    import ferrotherm as ft

    missing = missing_from_all(vars(ft), set(ft.__all__))
    if missing:
        print("public in ferrotherm but absent from __all__, so the stub cannot describe them:",
              file=sys.stderr)
        for n in missing:
            print(f"  {n}", file=sys.stderr)
        print("\nadd them to __all__ and rerun scripts/gen-stubs.py", file=sys.stderr)
        raise SystemExit(1)
    bare = unannotated_slots(vars(ft), set(ft.__all__))
    src = io.open("python/ferrotherm/__init__.py", encoding="utf-8").read()
    bare += undescribed_attributes(vars(ft), set(ft.__all__), src)
    if bare:
        print("public attributes with no annotation, so the stub cannot describe them:",
              file=sys.stderr)
        for n in bare:
            print(f"  {n}", file=sys.stderr)
        print("\nannotate them on the class and rerun scripts/gen-stubs.py", file=sys.stderr)
        raise SystemExit(1)
    print(f"__all__ covers every public name the module defines ({len(ft.__all__)} of them), and "
          "every public attribute they set is annotated")
