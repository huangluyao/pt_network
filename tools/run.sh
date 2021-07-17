#!/usr/bin/env bash

WORK_DIR="$PWD"
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

echo "$WORK_DIR/tools/train.py --config=$1"

python $WORK_DIR/tools/train.py --config=$1
