#!/usr/bin/env bash
set -x #echo on

source ./venv/bin/activate

gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir "/path/to/dir" \
    -- \
    --mode eval \
    --data_dir "/path/to/dir" \
    --vocab_path "/path/to/vocab.tsv" \
    --batch_size 128

deactivate
