#!/usr/bin/env bash

TRIAL_DIR="$( basename "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" )"
RUN_DIR="$( dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" )"
cd ${RUN_DIR}

# ---------------------------------------------------------------------------------------------
# baseline

stamp=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp} "${TRIAL_DIR}/config.yaml"
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp}-train/${stamp}-train-14.pt

swaify -b 96 -w 6 "${TRIAL_DIR}/swaify.yaml" herbarium.model.mobilenet_v3_large snapshots/${stamp}-train/${stamp}-train-1?.pt
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp}-train/${stamp}-train-14-swa.pt


# ---------------------------------------------------------------------------------------------
# label smoothing

stamp=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp} "${TRIAL_DIR}/config.yaml" label_smoothing=0.1
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp}-train/${stamp}-train-14.pt

stamp=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp} "${TRIAL_DIR}/config.yaml" label_smoothing=0.2
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp}-train/${stamp}-train-14.pt


# ---------------------------------------------------------------------------------------------
# three-fold ensemble

stamp0=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp0} "${TRIAL_DIR}/config.yaml" dset_nfolds=3 dset_vfold=0
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp0}-train/${stamp0}-train-14.pt

stamp1=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp1} "${TRIAL_DIR}/config.yaml" dset_nfolds=3 dset_vfold=1
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp1}-train/${stamp1}-train-14.pt

stamp2=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp2} "${TRIAL_DIR}/config.yaml" dset_nfolds=3 dset_vfold=2
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp2}-train/${stamp2}-train-14.pt

stamp=$(date +%s)
predict -b 96 -w 4 "${TRIAL_DIR}/predict-ensemble.yaml" snapshots/${stamp}-ensemble/${stamp}-0.pt \
                      weights_file_0=snapshots/${stamp0}-train/${stamp0}-train-14.pt \
                      weights_file_1=snapshots/${stamp1}-train/${stamp1}-train-14.pt \
                      weights_file_2=snapshots/${stamp2}-train/${stamp2}-train-14.pt \
                      reducer0=sum reducer1=sum samples_per_id=5


# ---------------------------------------------------------------------------------------------
# peak shift

stamp=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp} "${TRIAL_DIR}/config.yaml" peak_epoch=3 final_epoch=12
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp}-train/${stamp}-train-14.pt

stamp=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp} "${TRIAL_DIR}/config.yaml" peak_epoch=4 final_epoch=13
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp}-train/${stamp}-train-14.pt


# ---------------------------------------------------------------------------------------------
# group warmup

stamp=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp} "${TRIAL_DIR}/config.yaml" warmup_step=200 warmup_scale=0.1 peak_epoch=2 final_epoch=10
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp}-train/${stamp}-train-14.pt

stamp=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp} "${TRIAL_DIR}/config.yaml" warmup_step=350 warmup_scale=0.1 peak_epoch=2 final_epoch=10
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp}-train/${stamp}-train-14.pt

stamp=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp} "${TRIAL_DIR}/config.yaml" warmup_step=700 warmup_scale=0.1 peak_epoch=2 final_epoch=10
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp}-train/${stamp}-train-14.pt

stamp=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp} "${TRIAL_DIR}/config.yaml" warmup_step=350 warmup_scale=0.1 peak_epoch=4 final_epoch=12
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp}-train/${stamp}-train-14.pt

stamp=$(date +%s)
train -e 15 -b 96 -w 6 -r ${stamp} "${TRIAL_DIR}/config.yaml" warmup_step=700 warmup_scale=0.1 peak_epoch=4 final_epoch=12
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp}-train/${stamp}-train-14.pt


