from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
)
import pandas as pd


def print_metrics_from_df(df: pd.DataFrame, name: str) -> None:
    if len(df) == 0:
        print(f"{name}: N/A (empty df)")
        return

    y_true = df["y_true"]
    y_pred = df["y_pred"]

    if "pos_prob" not in df.columns:
        mcc = matthews_corrcoef(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print(f"{name} MCC: {mcc:.6f}")
        print(f"{name} Macro F1: {macro_f1:.6f}")
        print(f"{name} Weighted F1: {weighted_f1:.6f}")
        print(f"{name} Balanced Accuracy: {balanced_acc:.6f}")
        print(f"{name} Confusion Matrix:")
        print(cm)
        print(f"{name} Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        print()
        return

    y_prob = df["pos_prob"]

    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    auroc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"{name} MCC: {mcc:.6f}")
    print(f"{name} AUROC: {auroc:.6f}")
    print(f"{name} AUPR: {aupr:.6f}")
    print(f"{name} Precision: {precision:.6f}")
    print(f"{name} Recall: {recall:.6f}")
    print(f"{name} F1: {f1:.6f}")
    print(f"{name} Confusion Matrix:")
    print(cm)
    print(f"TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
    print()
