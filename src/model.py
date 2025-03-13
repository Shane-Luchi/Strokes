# src/model.py
from transformers import Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
import torch

def load_model(model_name='/home/LLMs/Qwen/Qwen2-VL-2B-Instruct', use_lora=True, lora_r=16, lora_alpha=32):
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name)
    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1
        )
        model = get_peft_model(model, lora_config)
    
    # Move to device safely
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model