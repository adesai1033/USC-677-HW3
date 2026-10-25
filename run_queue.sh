#!/bin/bash
cd "$(dirname "$0")/src" || exit 1
PY=../.venv/bin/python

echo "[queue] waiting for train_resnet.py to finish ($(date))"
while pgrep -f "train_resnet.py" > /dev/null; do sleep 120; done
echo "[queue] ResNet training done ($(date))"

$PY -u train_vit.py            2>&1 | tee ../outputs/logs/vit_full_train.log
$PY -u train_vit.py --freeze   2>&1 | tee ../outputs/logs/vit_frozen_train.log

$PY -u evaluate.py --model resnet                      2>&1 | tee ../outputs/logs/resnet18_eval.log
$PY -u evaluate.py --model vit --name vit_full         2>&1 | tee ../outputs/logs/vit_full_eval.log
$PY -u evaluate.py --model vit --name vit_frozen --freeze 2>&1 | tee ../outputs/logs/vit_frozen_eval.log

echo "[queue] all runs complete ($(date))"
