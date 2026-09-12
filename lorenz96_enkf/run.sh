#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")"

echo "=== Building lorenz96_enkf ==="
make

echo ""
echo "=== Running Lorenz-96 EnKF/LETKF twin experiment ==="
./lorenz96_enkf

echo ""
echo "=== Plotting diagnostics ==="
python3 plot_result.py