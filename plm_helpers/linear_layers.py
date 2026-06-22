import torch.nn as nn
from transformers.pytorch_utils import Conv1D

def get_lora_target_modules(model, exclude_lm_head=False, return_suffixes=True):
    names = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            if exclude_lm_head and name.endswith("lm_head"):
                continue
            names.append(name)

    if return_suffixes:
        names = sorted(set(n.split(".")[-1] for n in names))
    else:
        names = sorted(names)

    return names