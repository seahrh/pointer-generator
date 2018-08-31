#!/usr/bin/env bash
set -x #echo on

source ./venv/bin/activate

gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir /Users/me/googledrive/suma/log/huat \
    -- \
    --mode train \
    --data_path "/Users/my/googledrive/suma/finished_files/chunked/train_*" \
    --vocab_path /Users/my/googledrive/suma/finished_files/vocab \
    --max_step 30

deactivate
