import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CausalLMLabelOnlyCollator:
    tokenizer: Any
    label_id: int
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [
                {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
                for f in features
            ],
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]

        # labels pad
        labels_out = []
        for f in features:
            lab = f["labels"]
            pad_len = max_len - lab.shape[0]

            if pad_len > 0:
                lab = torch.cat(
                    [
                        lab,
                        torch.full(
                            (pad_len,), self.label_pad_token_id, dtype=torch.long
                        ),
                    ]
                )

            labels_out.append(lab)

        batch["labels"] = torch.stack(labels_out, dim=0)

        # mask prompt up to and including <LABEL>
        for i in range(batch["labels"].shape[0]):
            ids = batch["input_ids"][i]
            pos = (ids == self.label_id).nonzero(as_tuple=True)[0]

            if len(pos) == 0:
                continue

            j = int(pos[0])
            batch["labels"][i, : j + 1] = -100



        return batch
