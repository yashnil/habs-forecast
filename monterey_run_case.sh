#!/bin/bash
DATA_ROOT="${HABS_DATA_ROOT:-$HOME/Desktop/HABs_Research}"
MODEL_DIR="${HABS_MODEL_DIR:-$HOME/HAB_Models}"
FREEZE="${HABS_FREEZE:-$DATA_ROOT/Data/Derived/HAB_convLSTM_core_v1_clean.nc}"

python monterey.py \
  --obs "$FREEZE" \
  --pred "$MODEL_DIR/exports/convlstm__vanilla_best.nc:ConvLSTM" \
  --pred "$MODEL_DIR/exports/tft__convTFT_best.nc:TFT" \
  --pred "$MODEL_DIR/exports/pinn__convLSTM_best.nc:PINN" \
  --event-date 2021-05-25 \
  --bloom-lat 36.609 --bloom-lon -121.890 \
  --upsample 10 --smooth-sigma 1.4 \
  --output-dir bloom_panels