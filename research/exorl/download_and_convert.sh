#!/bin/bash

DOMAIN=${1:-walker}
ALGO=${2:-proto}
WORKERS=${3:-10}

./download.sh $DOMAIN $ALGO # takes ~ 1 min
python convert.py --num-workers $WORKERS --env-name $DOMAIN --expl-agent $ALGO # takes ~ 2 hours
