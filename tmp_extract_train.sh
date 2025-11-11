#!/usr/bin/env bash
set -euo pipefail
OUTDIR=/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/frames_train
IN=/home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_train.split1_cam01
mkdir -p 
mkdir -p /home/johnny/action_ws/output/otas_logs
LOG=/home/johnny/action_ws/output/otas_logs/step4_extract_train.log
: > 
processed=0; skipped=0; failed=0
shopt -s nullglob
for v in /*.avi; do
  bn=
  p=
  cam=; cam=
  rest=
  rest=
  dest=/__
  mkdir -p 
  if compgen -G /Frame_*.jpg > /dev/null; then
    echo [SKIP]
