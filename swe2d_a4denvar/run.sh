#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ " ${*} " == *" --help "* || " ${*} " == *" -h "* ]]; then
    exec python3 assimilate.py --help
fi
output_dir=$(python3 -c 'from assimilate import parser, validate_args; a=parser().parse_args(); validate_args(a); print(a.output)' "$@")
make test
python3 -u assimilate.py "$@"
python3 plot_result.py --output "$output_dir"
