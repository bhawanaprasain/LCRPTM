import torch.nn as nn

def print_trainable_parameters(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Trainable %: {100 * trainable / total:.4f}%")


def get_protgpt2_layers(model):
    """
    ProtGPT2 after PEFT wrapping:
        model.base_model.model.transformer.h

    Returns the GPT2 transformer block list.
    """
    return model.base_model.model.transformer.h


def unfreeze_bias_last_n_layers(model, n_last_layers):
    layers = get_protgpt2_layers(model)[-n_last_layers:]

    count = 0
    unfrozen = []

    for i, layer in enumerate(layers):
        for name, p in layer.named_parameters():
            if name.endswith("bias"):
                if not p.requires_grad:
                    p.requires_grad = True
                    count += p.numel()
                    unfrozen.append(f"layer_{len(get_protgpt2_layers(model)) - n_last_layers + i}.{name}")

    print(f"[Bias] Unfroze {count:,} parameters from last {n_last_layers} layers")
    if unfrozen:
        print("Unfrozen bias params:")
        for name in unfrozen:
            print(" ", name)

    return count


def is_norm_module(module):
    return isinstance(module, nn.LayerNorm)


def unfreeze_norm_last_n_layers(model, n_last_layers):
    layers = get_protgpt2_layers(model)[-n_last_layers:]

    count = 0
    unfrozen = []

    for i, layer in enumerate(layers):
        for module_name, module in layer.named_modules():
            if is_norm_module(module):
                for name, p in module.named_parameters(recurse=False):
                    if not p.requires_grad:
                        p.requires_grad = True
                        count += p.numel()
                        full_name = f"layer_{len(get_protgpt2_layers(model)) - n_last_layers + i}.{module_name}.{name}"
                        unfrozen.append(full_name)

    print(f"[Norm] Unfroze {count:,} parameters from last {n_last_layers} layers")
    if unfrozen:
        print("Unfrozen norm params:")
        for name in unfrozen:
            print(" ", name)

    return count