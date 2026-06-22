
import torch
import math


def compute_class_weights(df, label_col="Label"):
    n_pos = int((df[label_col] == 1).sum())
    n_neg = int((df[label_col] == 0).sum())

    # w_pos = n_neg / n_pos
    w_pos = math.sqrt(n_neg / n_pos)  
    

    w_neg = 1.0
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float)

    # print(f"Train counts Pos: {n_pos} Neg: {n_neg}")
    print(f"Class weights [NEG, POS]: {class_weights.tolist()}")

    return class_weights, n_pos, n_neg
