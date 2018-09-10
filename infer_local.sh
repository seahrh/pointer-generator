#!/usr/bin/env bash
set -x #echo on

source ./venv/bin/activate

gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir "/path/to/dir" \
    -- \
    --mode infer \
    --data_path "/path/to/val" \
    --vocab_path "/path/to/vocab.tsv" \

deactivate
