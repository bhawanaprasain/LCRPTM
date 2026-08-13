import os
import ast
import random
import numpy as np
import torch
import pandas as pd

from types import MethodType


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef

from peft import get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
)

from utils.config import *
from utils.settings import *
from utils.lora_config import *
from plm_helpers.dataset import PTMTrainDataset, PTMInferenceDataset
from plm_helpers.constants import (
    NEW_TOKENS,
    Token,
    PROTGPT2_END_TOKEN,
    PROGEN_PAD_TOKEN,
)
from plm_helpers.causal_lm_collator import CausalLMLabelOnlyCollator
from plm_helpers.embedding_patch import *
from plm_helpers.mask_emb import *
from plm_helpers.metrics import *


from plm_helpers.discriminative_trainer import DiscriminativeTrainer
from plm_helpers.token_utils import *
from plm_helpers.lm_head import *
from plm_helpers.prediction import evaluate_on_test
from plm_helpers.gpt2_inference import *
from utils.setup import set_seed, get_device
from utils.data import *
from utils.eval_utils import print_metrics_from_df
set_seed(seed)
DEVICE = get_device()

MODEL_NAME = f"nferruz/ProtGPT2"


# data = pd.read_csv(f"{train_df_path}")
# data["Label"] = data["Label"].astype(int)

# print("Label distribution (full):")
# print(data["Label"].value_counts())

# train_df, val_df = train_test_split(
#     data, test_size=validation_size, stratify=data["Label"], random_state=42
# )


import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(train_df_path)
data["Label"] = data["Label"].astype(int)

# print("Label distribution (full):")
# print(data["Label"].value_counts())
# if len(data) > 70000:
#     data, _ = train_test_split(
#         data,
#         train_size=70000,
#         # stratify=data["Label"],
#         random_state=seed
#     )

# print("\nSelected dataset size:", len(data))
# print("Label distribution (40K):")
# print(data["Label"].value_counts())

# -------------------------------------------------
# Train / validation split
# -------------------------------------------------
train_df, val_df = train_test_split(
    data,
    test_size=validation_size,
    stratify=data["Label"],
    random_state=seed
)

print("\nTrain size:", len(train_df))
print("Validation size:", len(val_df))

print("\nTrain distribution:")
print(train_df["Label"].value_counts())

print("\nValidation distribution:")
print(val_df["Label"].value_counts())

####################


print("Label distribution (sampled):")
print(train_df["Label"].value_counts())
print(val_df["Label"].value_counts())
class_weights, n_pos, n_neg = compute_class_weights(train_df)
print("Train counts  Pos:", n_pos, "Neg:", n_neg)
print("Class weights  [NEG, POS]:", class_weights.tolist())


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, pad_token=PROGEN_PAD_TOKEN
)
print("Tokenizer size before resizing:", len(tokenizer))
if motif_mode == "none":
    NEW_TOKENS = [
        t for t in NEW_TOKENS if t not in [Token.MARKER_START, Token.MARKER_END]
    ]
print(NEW_TOKENS, "new tokens to be added")
tokenizer.add_tokens(NEW_TOKENS)

token_ids = get_token_ids(tokenizer)

MARKER_START = token_ids.MARKER_START
MARKER_END = token_ids.MARKER_END
LABEL_ID = token_ids.LABEL_ID
POS_ID = token_ids.POS_ID
NEG_ID = token_ids.NEG_ID
SEQ_ID = token_ids.SEQ_ID

print("Tokenizer size after resizing:", len(tokenizer))

print(SEQ_ID, MARKER_START, MARKER_END, POS_ID, NEG_ID, LABEL_ID)




train_dataset = PTMTrainDataset(
    train_df,
    tokenizer,
    end_token=PROTGPT2_END_TOKEN,
    motif_mode=motif_mode,
    label_only_loss=True,
)

val_dataset = PTMTrainDataset(
    val_df,
    tokenizer,
    end_token=PROTGPT2_END_TOKEN,
    motif_mode=motif_mode,
    label_only_loss=True,
)

data_collator = CausalLMLabelOnlyCollator(tokenizer, LABEL_ID)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)


if hasattr(model, "lm_head"):

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_head):
        self.lm_head = new_head

    model.get_output_embeddings = MethodType(get_output_embeddings, model)
    model.set_output_embeddings = MethodType(set_output_embeddings, model)

model.resize_token_embeddings(len(tokenizer))
print(tokenizer.pad_token_id, "Pad token ID")
model.config.pad_token_id = tokenizer.pad_token_id
model.to(DEVICE)

model = get_peft_model(model, lora_config).to(DEVICE)


hidden_size = model.config.n_embd


        
base_emb = model.get_input_embeddings()
n_new = len(NEW_TOKENS)
old_vocab = base_emb.num_embeddings - n_new

model.set_input_embeddings(SplitTokenEmbedding(base_emb, old_vocab, n_new).to(DEVICE))

out_emb = model.get_output_embeddings()
hidden_size = getattr(model.config, "hidden_size", None) or getattr(
    model.config, "n_embd", None
)

# sanity: out_emb must have weight [vocab, hidden]
assert hasattr(out_emb, "weight"), "Output embedding/head has no weight"
assert (
    out_emb.weight.shape[0] == old_vocab + n_new
), "lm_head vocab mismatch after resize"
assert out_emb.weight.shape[1] == hidden_size, "lm_head hidden mismatch"

split_head = SplitLMHead(out_emb, hidden_size, old_vocab, n_new).to(DEVICE)
model.set_output_embeddings(split_head)



for name, param in model.get_input_embeddings().named_parameters():
    print("INPUT:", name, param.requires_grad, param.numel())



model.print_trainable_parameters()


output_dir = f"training_output/protgpt2/{ptm}/motif_marker_{motif_mode}_window_{max_seq_len}_target_modules_{target_modules}_lora_rank_{rank}_lora_alpha_{lora_alpha}_num_of_epochs_{num_of_epochs}"
output_dir = os.path.join(root, f"{output_dir}")

metrics_helper = CausalLMPTMMetrics(pos_id=POS_ID, neg_id=NEG_ID)

training_args = TrainingArguments(
    output_dir=f"{output_dir}",
    num_train_epochs=num_of_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    weight_decay=weight_decay,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model=metric_for_best_model,
    greater_is_better=greater_is_better,
    fp16=False,
    bf16=True,
    report_to=[],
    remove_unused_columns=False,
)

trainer = DiscriminativeTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    pos_id=POS_ID,
    neg_id=NEG_ID,
    class_weights=class_weights,
    compute_metrics=metrics_helper.compute_metrics,
    preprocess_logits_for_metrics=metrics_helper.preprocess_logits,
    margin=margin,
    margin_weight=margin_weight,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
    ],
)
trainer.train()

save_path = f"{output_dir}/lora"
os.makedirs(save_path, exist_ok=True)

trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

extra_state = {
    "old_vocab": old_vocab,
    "n_new": n_new,
    "input_embeddings": trainer.model.get_input_embeddings().state_dict(),
    "output_embeddings": trainer.model.get_output_embeddings().state_dict(),
}


torch.save(extra_state, os.path.join(save_path, "extra_modules.pt"))

torch.save(trainer.model.state_dict(), os.path.join(save_path, "full_model_state.pt"))

print(f"Saved to: {save_path}")
print(f"Saved extra modules  {os.path.join(save_path, 'extra_modules.pt')}")
print(f"Saved full state  {os.path.join(save_path, 'full_model_state.pt')}")



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
    trainer.model, test_loader, test_df, ["UniProtID", "pos"], POS_ID, NEG_ID, DEVICE
)
result_df.to_csv(f"{output_dir}/preds_heldout_test.csv")
print_metrics_from_df(result_df, "Held-out Test \n")

