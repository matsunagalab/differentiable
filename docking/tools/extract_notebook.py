"""
Extract Julia function definitions from train_param-apart.ipynb into a single
`.jl` file that can be included after docking.jl to form a canonical source.

Usage:
    python3 tools/extract_notebook.py > docking_canonical_overrides.jl

We extract code cells that define functions used by docking_score_elec (the
main 157-parameter learning entry point) and its rrule. Specifically cells
2, 4, 5, 6 of train_param-apart.ipynb.

Cell 60 (clipgrad!) and cell 62 (loss) are kept separately because they are
training-driver concerns, not scoring.

Notebooks were copy-pasted with full-width spaces (\u3000) in a few places;
we normalize those to ascii spaces so Julia parses cleanly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "train_param-apart.ipynb"

# Cells that define the *scoring* pipeline functions.
# Cell 3 is a debug print helper; we skip it.
SCORING_CELLS = [2, 4, 5, 6]

# Cells that define *training* helpers, kept separately so they don't leak
# into the scoring canonicalization.
TRAINING_CELLS = [60, 62]


def normalize(src: str) -> str:
    return src.replace("\u3000", " ")


def extract(indices: list[int]) -> str:
    nb = json.loads(NOTEBOOK.read_text())
    cells = nb["cells"]
    chunks: list[str] = []
    for idx in indices:
        cell = cells[idx]
        if cell["cell_type"] != "code":
            raise RuntimeError(f"cell {idx} is not a code cell")
        src = normalize("".join(cell["source"]))
        chunks.append(f"# ============ cell {idx} ============\n{src}")
    return "\n\n".join(chunks) + "\n"


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--training":
        sys.stdout.write(extract(TRAINING_CELLS))
    else:
        sys.stdout.write(extract(SCORING_CELLS))


if __name__ == "__main__":
    main()
