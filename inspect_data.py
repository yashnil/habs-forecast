

# Inspect the convlstm checkpoint
import torch, pprint
ck = torch.load("/Users/yashnilmohanty/HAB_Models/vanilla_best.pt", map_location="cpu")

# 1) Channel count the net expects (from the 1x1 'reduce' layer)
sd = ck.get("state_dict", ck)
print("reduce.weight:", sd["reduce.weight"].shape)  # -> torch.Size([24, 28, 1, 1])

# 2) Look for saved feature names / hparams
for k in ("hparams", "hyper_parameters", "config", "meta"):
    if k in ck: 
        print(k); pprint.pprint(ck[k])
for k in sd.keys():
    if "feature" in k or "feat" in k:
        print(k)
