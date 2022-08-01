#!/usr/bin/env bash

RUN_DIR="$( dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" )"
cd ${RUN_DIR}

# stamp=$(date +%s)
# train -e 15 -b 32 -w 2 -r ${stamp} 002-efficientnet-v2-s/config.yaml
# predict -b 32 -w 2 002-efficientnet-v2-s/predict.yaml snapshots/${stamp}-train/${stamp}-train-14.pt

# swaify -b 32 -w 4 002-efficientnet-v2-s/swaify.yaml herbarium.model.efficientnet_v2_s snapshots/${stamp}-train/${stamp}-train-1?.pt
# predict -b 32 -w 2 002-efficientnet-v2-s/predict.yaml snapshots/${stamp}-train/${stamp}-train-14-swa.pt


stamp=$(date +%s)
train -e 15 -b 32 -w 2 -r ${stamp} 002-efficientnet-v2-s/config.yaml peak_scale=10
predict -b 32 -w 2 002-efficientnet-v2-s/predict.yaml snapshots/${stamp}-train/${stamp}-train-14.pt
