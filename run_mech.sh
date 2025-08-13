#!/bin/bash
python mech_figure.py \
  --obs "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
  --pred "/Users/yashnilmohanty/HAB_Models/exports/convlstm__vanilla_best.nc:ConvLSTM" \
  --pred "/Users/yashnilmohanty/HAB_Models/exports/tft__convTFT_best.nc:TFT" \
  --pred "/Users/yashnilmohanty/HAB_Models/exports/pinn__convLSTM_best.nc:PINN" \
  --start 2019-01-01 --end 2021-06-30 \
  --out fig_robustness.png
