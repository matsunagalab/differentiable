#!/usr/bin/env bash
# Run all Julia reference/validation tests for the ZDOCK port.
# Invoke from the docking/ directory.

set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ ! -f docking_canonical_overrides_buggy.jl ]]; then
    echo ">>> regenerating docking_canonical_overrides_buggy.jl from the notebook"
    python3 tools/extract_notebook.py > docking_canonical_overrides_buggy.jl
fi

echo ">>> [1/2] sanity_check.jl"
julia tests/julia_ref/sanity_check.jl

echo ">>> [2/2] gradcheck.jl"
julia tests/julia_ref/gradcheck.jl

echo "All Julia reference tests passed."
