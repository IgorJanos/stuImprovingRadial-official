#!/usr/bin/env bash

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?
source ./docker/common.sh

DEVICE=$1
echo "Using GPU devices: ${DEVICE}"

DEVICE_NAME=`echo ${DEVICE} | tr "," "-"`

docker run \
    -it --rm \
    --name "stuImprovingRadial-${DEVICE_NAME}" \
    --gpus all \
    --privileged \
    --shm-size 24g \
    -v "${HOME}/.netrc":/root/.netrc \
    -v "${CWD}/..":/workspace/${PROJECT_NAME} \
    -v "/mnt/scratch/${USER}/.datasets/fiit":/mnt/datasets \
    -v "/mnt/persist/${USER}/${PROJECT_NAME}":/workspace/${PROJECT_NAME}/.mnt/persist \
    -v "/mnt/scratch/${USER}/${PROJECT_NAME}":/workspace/${PROJECT_NAME}/.mnt/scratch \
    -e CUDA_VISIBLE_DEVICES="${DEVICE}" \
    -e PYTHONPATH=/workspace/stuImprovingRadial \
    ${IMAGE_TAG}
