#!/usr/bin/env bash

set -eu

gcc -O2 swe2d_enkf_snsrf.c -lm -o da
./da
python3 plot_result.py
