
from enum import Enum
from plm_helpers.constants import Token, DiseaseToken


class MotifMode(str, Enum):
    NONE = "none"
    BPE_TOKEN = "bpe_token"
    CENTER_SPLIT = "center_split"


def build_label_text(label: int, end_token: str) -> str:
    label_tok = Token.POSITIVE.value if label == 1 else Token.NEGATIVE.value
    return f"{label_tok}{end_token}"

def build_disease_label_text(label: str, end_token: str) -> str:
    disease_to_token = {
        "Other": DiseaseToken.OTHER.value,
        "Lung adenocarcinoma": DiseaseToken.LUNG_ADENOCARCINOMA.value,
        "Head and neck squamous cell carcinoma": DiseaseToken.HEAD_AND_NECK_SQUAMOUS_CELL_CARCINOMA.value,
        "Skin cutaneous melanoma": DiseaseToken.SKIN_CUTANEOUS_MELANOMA.value,
        "Bladder urothelial carcinoma": DiseaseToken.BLADDER_UROTHELIAL_CARCINOMA.value,
        "Stomach adenocarcinoma": DiseaseToken.STOMACH_ADENOCARCINOMA.value,
    }

    label = "Other" if label is None else str(label).strip()
    label_tok = disease_to_token.get(label, DiseaseToken.OTHER.value)

    return f"{label_tok}{end_token}"


def build_prompt_ids(seq: str, tokenizer, motif_mode: str = "none"):
    """
    Returns prompt token ids for 3 ablations:

    1) none:
       SEQUENCE + BPE(seq) + LABEL

    2) bpe_token:
       SEQUENCE + BPE(seq) with motif markers inserted around the BPE token
       containing the center residue + LABEL

    3) center_split:
       SEQUENCE + BPE(left) + START + BPE(center residue only) + END + BPE(right) + LABEL
    """
    seq_prefix_ids = tokenizer(Token.SEQUENCE.value, add_special_tokens=False)[
        "input_ids"
    ]
    label_ids = tokenizer(Token.LABEL.value, add_special_tokens=False)["input_ids"]
    motif_start_ids = tokenizer(Token.MARKER_START.value, add_special_tokens=False)[
        "input_ids"
    ]
    motif_end_ids = tokenizer(Token.MARKER_END.value, add_special_tokens=False)[
        "input_ids"
    ]

    center_idx = len(seq) // 2

    # Case 1: no marker
    if motif_mode == MotifMode.NONE or motif_mode == "none":
        seq_ids = tokenizer(seq, add_special_tokens=False)["input_ids"]
        prompt_ids = seq_prefix_ids + seq_ids + label_ids
        return {
            "prompt_ids": prompt_ids,
            "mode": "none",
        }

    # Case 2: marker around full-sequence BPE token containing center residue
    if motif_mode == MotifMode.BPE_TOKEN or motif_mode == "bpe_token":
        enc = tokenizer(
            seq,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        seq_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        target_token_idx = None
        for i, (start, end) in enumerate(offsets):
            if start <= center_idx < end:
                target_token_idx = i
                break

        if target_token_idx is None:
            raise ValueError(
                f"Could not find token covering center residue at index {center_idx} for seq={seq}"
            )

        prompt_ids = (
            seq_prefix_ids
            + seq_ids[:target_token_idx]
            + motif_start_ids
            + [seq_ids[target_token_idx]]
            + motif_end_ids
            + seq_ids[target_token_idx + 1 :]
            + label_ids
        )

        return {
            "prompt_ids": prompt_ids,
            "mode": "bpe_token",
            "target_token_idx": target_token_idx,
        }

    # Case 3: split around center residue before tokenization
    if motif_mode == MotifMode.CENTER_SPLIT or motif_mode == "center_split":
        left = seq[:center_idx]
        center_res = seq[center_idx]
        right = seq[center_idx + 1 :]

        left_ids = tokenizer(left, add_special_tokens=False)["input_ids"]
        center_ids = tokenizer(center_res, add_special_tokens=False)["input_ids"]
        right_ids = tokenizer(right, add_special_tokens=False)["input_ids"]

        prompt_ids = (
            seq_prefix_ids
            + left_ids
            + motif_start_ids
            + center_ids
            + motif_end_ids
            + right_ids
            + label_ids
        )

        return {
            "prompt_ids": prompt_ids,
            "mode": "center_split",
            "center_residue": center_res,
        }

    raise ValueError(f"Unknown motif_mode: {motif_mode}")
