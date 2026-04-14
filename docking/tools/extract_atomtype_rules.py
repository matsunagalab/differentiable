"""Parse docking.jl `set_atomtype_id` and emit a Python-usable rule table.

The Julia function is ~370 elseif arms over (resname, atomname) → id,
plus a catch-all that assigns 12 (AILMV_mc) when A/I/L/M/V and 3 (mc) else.

We emit `src/zdock/_atomtype_rules.py` containing a tuple of
`(atomtype_id, resname, atomname)` rules. Run from `docking/`:

    python3 tools/extract_atomtype_rules.py
"""

from __future__ import annotations

import re
from pathlib import Path

JL = Path(__file__).resolve().parent.parent / "docking.jl"
OUT = (
    Path(__file__).resolve().parent.parent.parent
    / "docking_torch"
    / "src"
    / "zdock"
    / "_atomtype_rules.py"
)

PATTERN = re.compile(
    r"""(?x)
    ta\.resname\[iatom\]\s*==\s*"(?P<resname>[A-Z]+)"
    \s*&&\s*
    atomname\[iatom\]\s*==\s*"(?P<atomname>[A-Z0-9]+)"
    """
)
ASSIGN = re.compile(r"atomtype_id\[iatom\]\s*=\s*(?P<id>\d+)")


def main() -> None:
    src = JL.read_text().splitlines()

    # Locate the `function set_atomtype_id` block (the live one at the top
    # level, not the commented-out versions).
    in_func = False
    brace = 0
    rules: list[tuple[int, str, str]] = []
    pending_pair: tuple[str, str] | None = None
    for line in src:
        if not in_func and re.match(r"^\s*function\s+set_atomtype_id\b", line):
            in_func = True
            continue
        if not in_func:
            continue
        # End of function (first `^end` at column 0 after we entered).
        if re.match(r"^end\s*$", line):
            break

        mpat = PATTERN.search(line)
        if mpat:
            pending_pair = (mpat.group("resname"), mpat.group("atomname"))
            continue

        if pending_pair is not None:
            mass = ASSIGN.search(line)
            if mass:
                rules.append((int(mass.group("id")), *pending_pair))
                pending_pair = None

    # Emit the module.
    lines = [
        '"""Auto-generated atom-type rule table.',
        "",
        f"Parsed from {JL.relative_to(JL.parent.parent)} `set_atomtype_id`.",
        "Do not edit by hand — rerun tools/extract_atomtype_rules.py.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "# (atomtype_id, resname, atomname) tuples in the order they appear in",
        "# docking.jl's elseif ladder. Match first wins.",
        "ATOMTYPE_RULES: tuple[tuple[int, str, str], ...] = (",
    ]
    for (tid, r, a) in rules:
        lines.append(f"    ({tid}, {r!r}, {a!r}),")
    lines.append(")")
    lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} with {len(rules)} rules")


if __name__ == "__main__":
    main()
