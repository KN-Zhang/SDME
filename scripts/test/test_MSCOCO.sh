#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

DATASET_ROOT_PATH="/home/kz23d522/data/SDME/Dataset"
DATASET="MSCOCO"
CONFIG="configs/test.yaml"
CK_PATH="checkpoints/${DATASET}.pth"

python -m debugpy --listen vnode10:6666 --wait-for-client -m experiments.test \
        --gpuid=0 \
        --dataset_root_path=${DATASET_ROOT_PATH} \
        --dataset=${DATASET} \
        --config=${CONFIG} \
        --ck_path=${CK_PATH}
        
