import torch
import json
import numpy as np
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def torch_save_model(policy, model_path, cfg=None):
    torch.save({"state_dict": policy.state_dict(), "cfg": cfg}, model_path)
    with open(model_path.replace(".pth", ".json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)


def torch_load_model(model_path):
    model_dict = torch.load(model_path)
    cfg = None
    if "cfg" in model_dict:
        cfg = model_dict["cfg"]
    return model_dict["state_dict"], cfg

def to_onehot(tensor, num_class):
    x = torch.zeros(tensor.size() + (num_class,)).to(tensor.device)
    x.scatter_(-1, tensor.unsqueeze(-1), 1)
    return x
