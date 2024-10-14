#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

DATASET_ROOT_PATH="/home/kz23d522/data/SDME/Dataset"
DATASET="VIS_NIR"
CONFIG="configs/default.yaml"

python -m experiments.train \
        --gpuid=0 \
        --dataset_root_path=${DATASET_ROOT_PATH} \
        --dataset=${DATASET} \
        --config=${CONFIG}

