#!/usr/bin/env bash

WORK_DIR="$PWD"
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

echo "Usage: bash train.sh cfgs/xxx.json"

python $WORK_DIR/tools/train.py --config=$1
