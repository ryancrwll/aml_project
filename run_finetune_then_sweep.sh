#!/usr/bin/env bash
set -e

# Run finetune then grid sweep sequentially, logging to separate files
"$(pwd)/venv/bin/python" train_finetune_from_best.py 2>&1 | tee finetune_run.log
"$(pwd)/venv/bin/python" grid_sweep.py 2>&1 | tee grid_sweep.log
