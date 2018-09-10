#!/usr/bin/env bash
set -x #echo on

DATE=`date '+%Y%m%d_%H%M%S'`
BATCH_SIZE=64
BUCKET_NAME="mybucket"
EN_CORE_WEB_SM_PKG="gs://${BUCKET_NAME}/lib/en_core_web_sm-2.0.0.tar.gz"
STRINGX_PKG="gs://${BUCKET_NAME}/lib/sgcharts-stringx-1.1.1.tar.gz"
PACKAGES="${EN_CORE_WEB_SM_PKG},${STRINGX_PKG}"
MODEL_NAME="huat"
MODE="eval"
JOB_NAME="${MODEL_NAME}_${MODE}_${DATE}"
JOB_DIR="gs://${BUCKET_NAME}/models/${MODEL_NAME}"
DATA_DIR="gs://${BUCKET_NAME}/data/val"
VOCAB_PATH="gs://${BUCKET_NAME}/data/vocab.tsv"


gcloud ml-engine jobs submit training ${JOB_NAME} \
    --module-name trainer.task \
    --package-path trainer/ \
    --packages ${PACKAGES} \
    --job-dir ${JOB_DIR} \
    --config config.yaml \
    -- \
    --mode ${MODE} \
    --data_dir ${DATA_DIR} \
    --vocab_path ${VOCAB_PATH} \
    --batch_size ${BATCH_SIZE}
