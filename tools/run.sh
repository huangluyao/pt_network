#!/usr/bin/env bash

WORK_DIR="$PWD"
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

if  [ ! -n "$2" ] ;then
  echo "$WORK_DIR/tools/train.py --config=$1"
  python $WORK_DIR/tools/train.py --config=$1
else
  echo "$WORK_DIR/tools/train.py --config=$1 -p $2"
  python $WORK_DIR/tools/train.py --config=$1 -p $2
fi