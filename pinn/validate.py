# utils/validate.py
import torch, argparse, json
from pinn_convLSTM_v1p2 import make_loaders, ConvLSTM, ALL, rmse, DEVICE

parser = argparse.ArgumentParser()
parser.add_argument('--freeze', required=True, type=str)
parser.add_argument('--weights', required=True, type=str)
args = parser.parse_args()

tr_dl, va_dl, te_dl, st = make_loaders()
Cin = len([v for v in ALL if v in st])
net = ConvLSTM(Cin).to(DEVICE)
net.load_state_dict(torch.load(args.weights, map_location=DEVICE))

metrics = dict(
    train = rmse(tr_dl, net),
    val   = rmse(va_dl, net),
    test  = rmse(te_dl, net)
)
print(json.dumps(metrics, indent=2))

'''
{
  "train": 0.8544934828682454,
  "val": 0.8157419729476135,
  "test": 0.8264744185819607
}

'''