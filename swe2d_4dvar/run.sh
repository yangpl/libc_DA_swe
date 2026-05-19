#!/usr/bin/env bash

set -eu

gcc -O2 swe2d_4dvar.c -lm -o da
./da
python3 plot_4dvar_result.py
