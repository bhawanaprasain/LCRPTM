from peft import LoraConfig, TaskType
from peft import LoraConfig
from utils.config import *


lora_config = LoraConfig(
    r=rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules,
    bias="lora_only", # none, lora_only, all
    task_type=TaskType.CAUSAL_LM,
    fan_in_fan_out=False, # True for gpt2 False for progen

)


dora_config = LoraConfig(
    r=rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    use_dora=True,
    fan_in_fan_out=False, # True for gpt2 False for progen

)
