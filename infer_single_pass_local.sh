#!/usr/bin/env bash
set -x #echo on

source ./venv/bin/activate

gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir /Users/me/googledrive/suma/log/huat \
    -- \
    --mode infer \
    --data_path "/Users/me/googledrive/suma/finished_files/chunked/val_*" \
    --vocab_path /Users/me/googledrive/suma/finished_files/vocab \
    --single_pass 1

deactivate
