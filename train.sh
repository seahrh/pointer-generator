#!/usr/bin/env bash
set -x #echo on

DATE=`date '+%Y%m%d_%H%M%S'`
MAX_STEP=400
BUCKET_NAME="mybucket"
MODEL_NAME="mymodel"
JOB_NAME="${MODEL_NAME}_${DATE}"
JOB_DIR="gs://${BUCKET_NAME}/models/${MODEL_NAME}"
DATA_PATH="gs://${BUCKET_NAME}/data/finished_files/chunked/train_*"
VOCAB_PATH="gs://${BUCKET_NAME}/data/finished_files/vocab"


gcloud ml-engine jobs submit training ${JOB_NAME} \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir ${JOB_DIR} \
    --config config.yaml \
    -- \
    --mode train \
    --data_path ${DATA_PATH} \
    --vocab_path ${VOCAB_PATH} \
    --max_step ${MAX_STEP}
