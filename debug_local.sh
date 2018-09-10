#!/usr/bin/env bash
set -x #echo on

source ./venv/bin/activate

gcloud ml-engine local train \
    --module-name trainer.debug_dataset \
    --package-path trainer/ \
    -- \
    --data_path "/path/to/train" \
    --max_step 5 \
    --batch_size 2

deactivate
