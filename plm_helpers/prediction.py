import pandas as pd
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_on_test(
    model,
    dataloader,
    df,
    extra_cols,
    pos_id: int,
    neg_id: int,
    device: str,
):
    model.to(device)
    model.eval()

    rows = []
    row_offset = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        logits = outputs.logits

        # last non-pad token position in each sequence
        last_pos = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(logits.size(0), device=device)

        # logits at final prompt position predict next token
        class_logits = logits[batch_idx, last_pos][:, [neg_id, pos_id]]
        class_probs = F.softmax(class_logits, dim=-1)

        y_true = y.long()
        y_pred = (class_logits[:, 1] >= class_logits[:, 0]).long()

        class_logits = class_logits.cpu().numpy()
        class_probs = class_probs.cpu().numpy()
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        for i in range(len(y_true)):
            df_idx = row_offset + i

            record = {
                "row_index": int(df_idx),
                "y_true": int(y_true[i]),
                "y_pred": int(y_pred[i]),
                "neg_logit": float(class_logits[i, 0]),
                "pos_logit": float(class_logits[i, 1]),
                "neg_prob": float(class_probs[i, 0]),
                "pos_prob": float(class_probs[i, 1]),
                "Seq": df.iloc[df_idx]["Seq"],
            }

            if "average_score" in df.columns:
                record["average_score"] = df.iloc[df_idx]["average_score"]

            if extra_cols:
                for col in extra_cols:
                    if col in df.columns:
                        record[col] = df.iloc[df_idx][col]

            rows.append(record)

        row_offset += input_ids.size(0)

    out_df = pd.DataFrame(rows)
    return out_df