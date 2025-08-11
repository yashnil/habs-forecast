#!/bin/bash
python case_studies_individual.py \
  --obs "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
  --pred "/Users/yashnilmohanty/HAB_Models/exports/convlstm__vanilla_best.nc:ConvLSTM" \
  --pred "/Users/yashnilmohanty/HAB_Models/exports/tft__convTFT_best.nc:TFT" \
  --pred "/Users/yashnilmohanty/HAB_Models/exports/pinn__convLSTM_best.nc:PINN" \
  --event-date 2021-05-25 \
  --bloom-lat 36.609 --bloom-lon -121.890 \
  --upsample 10 --smooth-sigma 1.4 \
  --output-dir bloom_panels