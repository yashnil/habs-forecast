#!/bin/bash
# Run diagnostics for optimized PINN model

FREEZE="/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc"
CKPT="$HOME/HAB_Models/convLSTM_best.pt"
OUT="Diagnostics_PINN_Optimized"

echo "========================================="
echo "Running PINN Diagnostics"
echo "========================================="
echo "Freeze file: $FREEZE"
echo "Checkpoint:  $CKPT"
echo "Output dir:  $OUT"
echo "========================================="
echo ""

if [ ! -f "$FREEZE" ]; then
    echo "ERROR: Freeze file not found: $FREEZE"
    exit 1
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

python pinn/diagnostics.py \
    --freeze "$FREEZE" \
    --ckpt   "$CKPT" \
    --out    "$OUT" \
    --seq 6 --lead 1 --batch 32

echo ""
echo "========================================="
echo "Diagnostics complete!"
echo "Results saved to: $OUT"
echo "========================================="

