#!/bin/bash

CONFIG=utils/configs.json

export PYTHONPATH=$pwd:$PYTHONPATH


window_size=$(python3 -c "import json; print(json.load(open('$CONFIG'))['max_seq_len'])")
rank=$(python3 -c "import json; print(json.load(open('$CONFIG'))['rank'])")
progen_model_size=$(python3 -c "import json; print(json.load(open('$CONFIG'))['progen_model_size'])")

ptm_alias=$(python3 -c "import json; print(json.load(open('$CONFIG'))['ptm_alias'])")
lora_alpha=$(python3 -c "import json; print(json.load(open('$CONFIG'))['lora_alpha'])")
num_of_epochs=$(python3 -c "import json; print(json.load(open('$CONFIG'))['num_of_epochs'])")
motif_mode=$(python3 -c "import json; print(json.load(open('$CONFIG'))['motif_mode'])")

target_modules=$(python3 -c "import json; cfg=json.load(open('$CONFIG')); print(''.join(cfg['target_modules']))")
echo "$target_modules"








mkdir -p training_logs

mkdir -p training_logs/protgpt2/${ptm_alias}

nohup python3 -u src/protgpt2/lora_ft.py > training_logs/protgpt2/${ptm_alias}/motif_marker_${motif_mode}_window_${window_size}_target_modules_${target_modules}_lora_rank_${rank}_lora_alpha_${lora_alpha}_num_of_epochs_${num_of_epochs}.log 2>&1 &

