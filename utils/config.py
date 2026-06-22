import json
import os
import ast


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


root = get_project_root()
config_dir = os.path.join(root, "utils")
config_path = os.path.join(config_dir, "configs.json")


with open(config_path, "r") as file:
    config = json.load(file)
    print("Training config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

rank = config["rank"]
seed = config["seed"]
max_seq_len = config["max_seq_len"]
data_dir = config["data_dir"]
logs_dir = config["logs_dir"]
ptm = config["ptm_alias"]
lora_alpha = config["lora_alpha"]
lora_dropout = config["lora_dropout"]
learning_rate = config["learning_rate"]
fan_in_fan_out = config["fan_in_fan_out"]
num_of_epochs = config["num_of_epochs"]
target_modules = config["target_modules"]
motif_mode = config["motif_mode"]
margin = config["margin"]
margin_weight = config["margin_weight"]
warmup_ratio = config["warmup_ratio"]
weight_decay = config["weight_decay"]
label_smoothing = config["label_smoothing"]
neftune_noise_alpha = config["neftune_noise_alpha"]
metric_for_best_model = config["metric_for_best_model"]
greater_is_better = config["greater_is_better"]
validation_size = config["validation_size"]
progen_model_size = config["progen_model_size"]
training_output_dir = config["training_output_dir"]
early_stopping_patience = config["early_stopping_patience"]
early_stopping_threshold = config["early_stopping_threshold"]
per_device_train_batch_size = config["per_device_train_batch_size"]
per_device_eval_batch_size = config["per_device_eval_batch_size"]
gradient_accumulation_steps = config["gradient_accumulation_steps"]

target_residue = ptm.split("_")[-1].upper()
