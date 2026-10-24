#!/usr/bin/env bash
# Build, validate, assimilate, and plot the binomial checkpointing model.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ " ${*} " == *" --help "* || " ${*} " == *" -h "* ]]; then
    exec python3 assimilate.py --help
fi
# Parse before running tests so invalid command-line options fail immediately.
output_dir=$(python3 -c 'from assimilate import parser, validate_args; a = parser().parse_args(); validate_args(a); print(a.output)' "$@")
make test
python3 -u assimilate.py "$@"
python3 plot_4dvar_result.py --output "$output_dir" --movie
