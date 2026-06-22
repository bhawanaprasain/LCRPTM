import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, roc_auc_score


class CausalLMPTMMetrics:
    def __init__(self, pos_id: int, neg_id: int):
        self.pos_id = pos_id
        self.neg_id = neg_id

    def preprocess_logits(self, logits, labels):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        mask = (labels == self.pos_id) | (labels == self.neg_id)
        has = mask.any(dim=1)

        t = mask.float().argmax(dim=1)
        t_prev = torch.clamp(t - 1, min=0)
        t_prev = torch.where(has, t_prev, torch.zeros_like(t_prev))

        b = torch.arange(logits.size(0), device=logits.device)

        two = logits[b, t_prev][:, [self.neg_id, self.pos_id]]
        return two

    # def compute_metrics(self, eval_pred):
    #     two, labels = eval_pred

    #     two = np.asarray(two)
    #     labels = np.asarray(labels)

    #     y_true = []

    #     for i in range(labels.shape[0]):
    #         row = labels[i]
    #         pos = np.where((row == self.pos_id) | (row == self.neg_id))[0]

    #         if len(pos) == 0:
    #             continue

    #         t = int(pos[0])
    #         y_true.append(1 if row[t] == self.pos_id else 0)

    #     if len(y_true) == 0:
    #         return {"mcc": 0.0}

    #     y_pred = (two[:, 1] >= two[:, 0]).astype(int)
    #     y_pred = y_pred[: len(y_true)]
            
    #     mcc = matthews_corrcoef(y_true, y_pred)
    #     f1 = f1_score(y_true, y_pred)

    #     return {
    #         "mcc": float(mcc),
    #         "f1": float(f1),
    #     }
    def compute_metrics(self, eval_pred):
        two, labels = eval_pred

        two = np.asarray(two)
        labels = np.asarray(labels)

        y_true = []

        for i in range(labels.shape[0]):
            row = labels[i]
            pos = np.where((row == self.pos_id) | (row == self.neg_id))[0]

            if len(pos) == 0:
                continue

            t = int(pos[0])
            y_true.append(1 if row[t] == self.pos_id else 0)

        if len(y_true) == 0:
            return {"mcc": 0.0, "f1": 0.0, "auroc": 0.0}

        y_true = np.asarray(y_true)

        two = two[: len(y_true)]

        scores = two[:, 1] - two[:, 0]   # positive confidence score
        y_pred = (scores >= 0).astype(int)

        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # AUROC needs both classes present
        if len(np.unique(y_true)) < 2:
            auroc = 0.0
        else:
            auroc = roc_auc_score(y_true, scores)
        pos_scores = scores[y_true == 1]
        neg_scores = scores[y_true == 0]
        return {
            "mcc": float(mcc),
            "f1": float(f1),
            "auroc": float(auroc),
            "pos_margin_mean": float(pos_scores.mean()) if len(pos_scores) else 0.0,
            "neg_margin_mean": float(neg_scores.mean()) if len(neg_scores) else 0.0,
            "margin_gap": float(pos_scores.mean() - neg_scores.mean())
                if len(pos_scores) and len(neg_scores) else 0.0,

            "pos_frac_above_0": float((pos_scores >= 0).mean()) if len(pos_scores) else 0.0,
            "neg_frac_below_0": float((neg_scores < 0).mean()) if len(neg_scores) else 0.0,
        }


import torch
import numpy as np

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report,
)


class DiseaseMetrics:
    def __init__(
        self,
        disease_token_ids,
        label_token_id,
    ):
        self.disease_token_ids = disease_token_ids
        self.label_token_id = label_token_id

    def preprocess_logits(self, logits, labels):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        disease_ids = torch.tensor(
            self.disease_token_ids,
            device=logits.device,
            dtype=torch.long,
        )

        mask = (labels.unsqueeze(-1) == disease_ids).any(dim=-1)
        has = mask.any(dim=1)

        t = mask.long().argmax(dim=1)
        t_prev = torch.clamp(t - 1, min=0)
        t_prev = torch.where(has, t_prev, torch.zeros_like(t_prev))

        b = torch.arange(logits.size(0), device=logits.device)

        disease_logits = logits[b, t_prev][:, self.disease_token_ids]

        return disease_logits

    def compute_metrics(self, eval_pred):
        disease_logits, labels = eval_pred

        disease_logits = np.asarray(disease_logits)
        labels = np.asarray(labels)

        tok_to_class = {
            tok: idx
            for idx, tok in enumerate(self.disease_token_ids)
        }

        disease_ids = np.asarray(self.disease_token_ids)

        y_true = []
        valid_indices = []

        for i, row in enumerate(labels):
            pos = np.where(np.isin(row, disease_ids))[0]

            if len(pos) == 0:
                continue

            t = int(pos[0])
            disease_tok = row[t]

            if disease_tok not in tok_to_class:
                continue

            y_true.append(tok_to_class[disease_tok])
            valid_indices.append(i)

        if len(y_true) == 0:
            return {
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "balanced_acc": 0.0,
            }

        y_true = np.asarray(y_true)

        disease_logits = disease_logits[valid_indices]

        y_pred = disease_logits.argmax(axis=1)

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        balanced_acc = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        return {
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "balanced_acc": float(balanced_acc),
        }
