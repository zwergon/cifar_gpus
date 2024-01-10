#!/bin/bash

#MSUB -r lightning_train
#MSUB -n 1
#MSUB -c 128
#MSUB -T 7200
#MSUB -q a100
#MSUB -A ifp00083@A100
#MSUB -m scratch,store

DATA_DIR=${CCCSCRATCHDIR}/lecomtje/data
SRC_DIR=${CCCSTOREDIR}/repositories/cifar_gpus
export PYTHONPATH=${SRC_DIR}:$PYTHONPATH

source ${CCCSCRATCHDIR}/lecomtje/venvs/drp3d/bin/activate
env PATH_DATASETS=${DATA_DIR}/cifar10 python ${SRC_DIR}/cifar_gpus/lightning/main.py
