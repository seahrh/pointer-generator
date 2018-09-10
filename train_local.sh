#!/usr/bin/env bash
set -x #echo on

source ./venv/bin/activate

gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir "/path/to/dir" \
    -- \
    --mode train \
    --data_dir "/path/to/train" \
    --vocab_path "/path/to/vocab.tsv" \
    --max_step 61

deactivate
