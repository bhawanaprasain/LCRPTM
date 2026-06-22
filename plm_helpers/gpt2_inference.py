import torch


def pad_collate(batch, pad_id: int):
    max_len = max(x["input_ids"].numel() for x in batch)

    input_ids, attention_mask, ys = [], [], []
    raw_seqs, prompts = [], []

    for x in batch:
        ids = x["input_ids"]
        am = x["attention_mask"]
        pad_len = max_len - ids.numel()

        if pad_len > 0:
            ids = torch.cat(
                [ids, torch.full((pad_len,), pad_id, dtype=torch.long)], dim=0
            )
            am = torch.cat([am, torch.zeros((pad_len,), dtype=torch.long)], dim=0)

        input_ids.append(ids)
        attention_mask.append(am)
        ys.append(x["labels"])
        raw_seqs.append(x["raw_seq"])
        prompts.append(x["prompt"])

    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "labels": torch.stack(ys, dim=0),
        "raw_seq": raw_seqs,
        "prompt": prompts,
    }
