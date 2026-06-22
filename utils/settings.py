import os
from utils.config import *


def create_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


logs_dir = os.path.join(root, logs_dir)
# data_dir = os.path.join(root, data_dir)

training_output_dir = os.path.join(root, training_output_dir)
adapter_dir = os.path.join(training_output_dir, f"{ptm}/lora_adapter")
classifier_head_dir = os.path.join(training_output_dir, f"{ptm}/classifier_head")
classifier_head_config_dir = os.path.join(
    training_output_dir, f"{ptm}/classifier_head_config"
)
results_dir = os.path.join(training_output_dir, f"{ptm}/results")

train_df_path = os.path.join(data_dir, f"{ptm}/train.csv")
test_df_path = os.path.join(data_dir, f"{ptm}/test.csv")
val_df_path =  os.path.join(data_dir, f"{ptm}/validation.csv")
# train_df_path = os.path.join(data_dir, f"{ptm}/train.csv")
# test_df_path = os.path.join(data_dir, f"{ptm}/test.csv")

param_combination = f"window_size_{max_seq_len}_target_modules_{''.join(target_modules)}__lora_{rank}_alpha_{lora_alpha}"
peft_model_path = os.path.join(adapter_dir, f"{param_combination}")
classifier_state_dict_path = os.path.join(
    classifier_head_dir, f"{param_combination}.pt"
)
classifier_config_path = os.path.join(
    classifier_head_config_dir, f"{param_combination}_config.json"
)
results_csv_path = os.path.join(results_dir, f"{param_combination}.csv")
results_json_path = os.path.join(results_dir, f"{param_combination}.json")

# create_directory(logs_dir)
# create_directory(training_output_dir)
# create_directory(adapter_dir)
# create_directory(classifier_head_dir)
# create_directory(classifier_head_config_dir)
# create_directory(results_dir)
