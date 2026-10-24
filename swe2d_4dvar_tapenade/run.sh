#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Build and test the generated C adjoint; validate its source manifest.
make test
python3 -u assimilate.py "$@"
# Find the chosen output path using the same argument parser as the experiment.
output_dir=$(python3 -c 'from assimilate import parser; print(parser().parse_args().output)' "$@")
python3 plot_4dvar_result.py --output "$output_dir" --movie
