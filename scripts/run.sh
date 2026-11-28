#!/bin/bash
# usage: run.sh <cmd> [args]
cd ~/vlia-nha
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 HF_HOME=/data/hf_cache
exec systemd-inhibit --what=sleep:idle --who=stage23 --why="stage23 $1" nice -n 10 .venv/bin/python -I -u breakfast/scripts/stage23_breakfast.py "$@"
