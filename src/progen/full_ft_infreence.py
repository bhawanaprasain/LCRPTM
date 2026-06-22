from utils.config import *
from utils.settings import *
from utils.setup import set_seed, get_device

from plm_helpers.token_utils import *
from plm_helpers.constants import NEW_TOKENS
from plm_helpers.dataset import PTMInferenceDataset
from plm_helpers.gpt2_inference import pad_collate
from plm_helpers.constants import NEW_TOKENS, Token, PROTGPT2_END_TOKEN
from plm_helpers.prediction import evaluate_on_test
from plm_helpers.embedding_patch import *

import pandas as pd
from sklearn.metrics import matthews_corrcoef

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from types import MethodType

set_seed(seed)
DEVICE = get_device()
base_model_id = f"hugohrban/progen2-{progen_model_size}"
if motif_mode == "none":
    NEW_TOKENS = [
        t for t in NEW_TOKENS if t not in [Token.MARKER_START, Token.MARKER_END]
    ]


output_dir = f"training_output/full_finetuned_progen_{progen_model_size}/{ptm}/motif_marker_{motif_mode}_window_{max_seq_len}_num_of_epochs_{num_of_epochs}"
output_dir = os.path.join(root, f"{output_dir}/full_model")
print(os.path.exists(output_dir))
tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)

# New tokens should exist in saved dir
for t in NEW_TOKENS:
    if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id:
        raise RuntimeError(
            f"Tokenizer at {output_dir} is missing token {t}. "
            f"Fix training save: tokenizer.save_pretrained"
        )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

token_ids = get_token_ids(tokenizer)

MARKER_START = token_ids.MARKER_START
MARKER_END = token_ids.MARKER_END
LABEL_ID = token_ids.LABEL_ID
POS_ID = token_ids.POS_ID
NEG_ID = token_ids.NEG_ID
SEQ_ID = token_ids.SEQ_ID


model = AutoModelForCausalLM.from_pretrained(
    f"hugohrban/progen2-{progen_model_size}", trust_remote_code=True
)

embedding_path, embedding_module = find_token_embedding_module(model)

if embedding_module is None:
    raise RuntimeError(
        "Could not find token embedding module in base model. "
        "Print(model) and add the correct embedding path."
    )

model.get_input_embeddings = MethodType(
    make_get_input_embeddings(embedding_path), model
)
model.set_input_embeddings = MethodType(
    make_set_input_embeddings(embedding_path), model
)

if hasattr(model, "lm_head"):

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_head):
        self.lm_head = new_head

    model.get_output_embeddings = MethodType(get_output_embeddings, model)
    model.set_output_embeddings = MethodType(set_output_embeddings, model)

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = len(tokenizer)
state = torch.load(f"{output_dir}/pytorch_model.bin", map_location="cpu")
missing, unexpected = model.load_state_dict(state, strict=False)
ignore_suffixes = (".attn.bias", ".masked_bias")
bad_missing = [k for k in missing if not k.endswith(ignore_suffixes)]
if len(bad_missing) or len(unexpected):
    print("\n[ERROR] final_model did not load cleanly.")
    print("Missing keys (first 50):", bad_missing[:50])
    print("Unexpected keys (first 50):", unexpected[:50])
    raise RuntimeError("State dict mismatch -> MCC may differ. Fix saving/loading.")

test_df = pd.read_csv(test_df_path)
test_dataset = PTMInferenceDataset(
    test_df,
    tokenizer,
    motif_mode=motif_mode,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=lambda b: pad_collate(b, tokenizer.pad_token_id),
)


result_df = evaluate_on_test(
    model,
    test_loader,
    test_df,
    ["UniProtID", "pos"],
    POS_ID,
    NEG_ID,
    DEVICE,
)
mcc = matthews_corrcoef(result_df["y_true"], result_df["y_pred"])
print(f" MCC: {mcc:.6f}")
