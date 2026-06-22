import os
import torch
import torch.nn as nn
import pandas as pd

from types import MethodType
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.config import *
from utils.settings import *
from utils.setup import set_seed, get_device

from plm_helpers.token_utils import *
from plm_helpers.constants import NEW_TOKENS, Token
from plm_helpers.dataset import PTMInferenceDataset
from plm_helpers.gpt2_inference import pad_collate
from plm_helpers.prediction import evaluate_on_test
from plm_helpers.embedding_patch import *
from plm_helpers.lm_head import *

set_seed(seed)
DEVICE = get_device()

base_model_id = "nferruz/ProtGPT2"

if motif_mode == "none":
    NEW_TOKENS = [
        t for t in NEW_TOKENS if t not in [Token.MARKER_START, Token.MARKER_END]
    ]

model_name = "protgpt2"

output_dir = (
    f"training_output/{model_name}/{ptm}/motif_marker_{motif_mode}"
    f"_window_{max_seq_len}_target_modules_{target_modules}"
    f"_lora_rank_{rank}_lora_alpha_{lora_alpha}"
    f"_num_of_epochs_{num_of_epochs}"
)

LORA_DIR = os.path.join(root, f"{output_dir}/lora")
EXTRA_PATH = os.path.join(LORA_DIR, "extra_modules.pt")
FULL_STATE_PATH = os.path.join(LORA_DIR, "full_model_state.pt")

print("LORA_DIR:", LORA_DIR)
print("Exists:", os.path.exists(LORA_DIR))
print("Extra exists:", os.path.exists(EXTRA_PATH))
print("Full state exists:", os.path.exists(FULL_STATE_PATH))

if not os.path.exists(LORA_DIR):
    raise FileNotFoundError(f"Missing LORA_DIR: {LORA_DIR}")

if not os.path.exists(EXTRA_PATH):
    raise FileNotFoundError(f"Missing extra_modules.pt: {EXTRA_PATH}")

if not os.path.exists(FULL_STATE_PATH):
    raise FileNotFoundError(f"Missing full_model_state.pt: {FULL_STATE_PATH}")


tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, trust_remote_code=True)

for t in NEW_TOKENS:
    if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id:
        raise RuntimeError(
            f"Tokenizer at {LORA_DIR} is missing token {t}. "
            f"Fix training save: tokenizer.save_pretrained(LORA_DIR)"
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

print("Token IDs:")
print("SEQ_ID:", SEQ_ID)
print("MARKER_START:", MARKER_START)
print("MARKER_END:", MARKER_END)
print("POS_ID:", POS_ID)
print("NEG_ID:", NEG_ID)
print("LABEL_ID:", LABEL_ID)
print("pad_token_id:", tokenizer.pad_token_id)
print("tokenizer size:", len(tokenizer))


# load saved extras
extra_state = torch.load(EXTRA_PATH, map_location="cpu")
old_vocab = extra_state["old_vocab"]
n_new = extra_state["n_new"]

print("old_vocab:", old_vocab)
print("n_new:", n_new)


model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
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


base_emb = model.get_input_embeddings()
model.set_input_embeddings(
    SplitTokenEmbedding(base_emb, old_vocab, n_new)
)


out_emb = model.get_output_embeddings()
hidden_size = getattr(model.config, "hidden_size", None) or getattr(
    model.config, "n_embd", None
)

if hidden_size is None:
    raise RuntimeError("Could not infer hidden size from model.config")

model.set_output_embeddings(
    SplitLMHead(out_emb, hidden_size, old_vocab, n_new)
)


model.get_input_embeddings().load_state_dict(
    extra_state["input_embeddings"],
    strict=True,
)
model.get_output_embeddings().load_state_dict(
    extra_state["output_embeddings"],
    strict=True,
)

print("Loaded input embedding state.")
print("Loaded output embedding state.")



model = PeftModel.from_pretrained(model, LORA_DIR)


full_state = torch.load(FULL_STATE_PATH, map_location="cpu")
missing_keys, unexpected_keys = model.load_state_dict(full_state, strict=False)

print("Loaded full model state.")
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Input embedding type:", type(model.get_input_embeddings()))
print("Output embedding type:", type(model.get_output_embeddings()))


test_df = pd.read_csv(f"{data_dir}/{ptm}/test.csv")

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
print(f"Test MCC: {mcc:.6f}")
