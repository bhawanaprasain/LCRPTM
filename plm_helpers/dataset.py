

import pandas as pd
import torch
from torch.utils.data import Dataset

from plm_helpers.input_formatting import build_prompt_ids, build_label_text, build_disease_label_text


class BasePTMDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        motif_mode: str = "none",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.motif_mode = motif_mode

    def __len__(self):
        return len(self.df)

    def _get_seq(self, idx: int) -> str:
        return str(self.df.loc[idx, "Seq"])

    def _get_label(self, idx: int) -> int:
        if "Label" not in self.df.columns:
            return -1
        return int(self.df.loc[idx, "Label"])

    def _build_prompt(self, seq: str):
        prompt_obj = build_prompt_ids(
            seq=seq,
            tokenizer=self.tokenizer,
            motif_mode=self.motif_mode,
        )
        prompt_ids = prompt_obj["prompt_ids"]

        prompt_text = self.tokenizer.decode(
            prompt_ids,
            clean_up_tokenization_spaces=False,
        )
        return prompt_text, prompt_ids, prompt_obj


class PTMTrainDataset(BasePTMDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        end_token: str,
        motif_mode: str = "none",
        label_only_loss: bool = True,
    ):
        super().__init__(
            df=df,
            tokenizer=tokenizer,
            motif_mode=motif_mode,
        )
        self.end_token = end_token
        self.label_only_loss = label_only_loss

    def __getitem__(self, idx):
        seq = self._get_seq(idx)
        y = self._get_label(idx)

        prompt_text, prompt_ids, prompt_obj = self._build_prompt(seq)

        label_text = build_label_text(y, self.end_token)
        label_ids = self.tokenizer(
            label_text,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = torch.tensor(prompt_ids + label_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        if self.label_only_loss:
            labels = torch.full_like(input_ids, -100)
            labels[len(prompt_ids) :] = input_ids[len(prompt_ids) :]
        else:
            labels = input_ids.clone()

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "raw_seq": seq,
            "prompt": prompt_text,
            "motif_mode": self.motif_mode,
        }


        return sample


class PTMInferenceDataset(BasePTMDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        motif_mode: str = "none",
    ):
        super().__init__(
            df=df,
            tokenizer=tokenizer,
            motif_mode=motif_mode,
        )

    def __getitem__(self, idx):
        seq = self._get_seq(idx)
        y = self._get_label(idx)

        prompt_text, prompt_ids, prompt_obj = self._build_prompt(seq)

        return {
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long),
            "raw_seq": seq,
            "prompt": prompt_text,
            "motif_mode": self.motif_mode,
        }




class DiseaseTrainDataset(BasePTMDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        end_token: str,
        motif_mode: str = "none",
        label_only_loss: bool = True,
    ):
        super().__init__(
            df=df,
            tokenizer=tokenizer,
            motif_mode=motif_mode,
        )
        self.end_token = end_token
        self.label_only_loss = label_only_loss

    def _get_disease(self, idx: int) -> str:
        if "Disease" not in self.df.columns:
            return "Other"
        return str(self.df.loc[idx, "Disease"])

    def __getitem__(self, idx):
        seq = self._get_seq(idx)
        y = self._get_disease(idx)

        prompt_text, prompt_ids, prompt_obj = self._build_prompt(seq)

        label_text = build_disease_label_text(y, self.end_token)
        label_ids = self.tokenizer(
            label_text,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = torch.tensor(prompt_ids + label_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        if self.label_only_loss:
            labels = torch.full_like(input_ids, -100)
            labels[len(prompt_ids) :] = input_ids[len(prompt_ids) :]
        else:
            labels = input_ids.clone()

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "raw_seq": seq,
            "prompt": prompt_text,
            "motif_mode": self.motif_mode,
        }

        return sample


class DiseaseInferenceDataset(BasePTMDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        motif_mode: str = "none",
    ):
        super().__init__(
            df=df,
            tokenizer=tokenizer,
            motif_mode=motif_mode,
        )

    def _get_disease(self, idx: int) -> str:
        if "Disease" not in self.df.columns:
            return "Other"
        return str(self.df.loc[idx, "Disease"])

    def __getitem__(self, idx):
        seq = self._get_seq(idx)
        y = self._get_disease(idx)

        prompt_text, prompt_ids, prompt_obj = self._build_prompt(seq)

        return {
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
            "labels": torch.tensor(0, dtype=torch.long),
            "raw_seq": seq,
            "prompt": prompt_text,
            "motif_mode": self.motif_mode,
        }
