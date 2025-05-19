'''

CUDA_VISIBLE_DEVICES=5 nohup python 111.py > logs/inference_levenshtein_lora_515.log 2>&1 &

'''

import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import json
import time
import os
import random
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from Levenshtein import distance as levenshtein_distance

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ====================== 新增EOS定义 ======================
EOS_TOKEN = "<EOS>"

# 加载 tokenizer 和 processor
tokenizer = AutoTokenizer.from_pretrained(
    "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/", use_fast=False, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("/home/LLMs/Qwen/Qwen2-VL-2B-Instruct")

# 添加 EOS Token 到 tokenizer
tokenizer.add_special_tokens({'eos_token': EOS_TOKEN})
EOS_TOKEN_ID = tokenizer.eos_token_id





# ===================== 推理函数 =====================
def predict(messages, model, max_new_tokens=64, temperature=1.0, top_p=0.9, top_k=50):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=EOS_TOKEN_ID,    # 指定终止符号
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature,      # 控制生成的随机性，多样性
        top_p=top_p,                  # nucleus sampling
        top_k=top_k                   # 限制采样范围
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]

# 加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.resize_token_embeddings(len(tokenizer))  # 🚨 调整Embedding以适配新token

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    inference_mode=True,
    r=16,
    lora_alpha=32, #一般是2r
    lora_dropout=0.05,
    bias="none",
)



print("--------------starting to test---------------")

checkpoint_path = "./output_levenshtein_lora/Qwen2-VL-2B/checkpoint-11000"  # 确保路径正确
val_peft_model = PeftModel.from_pretrained(model, checkpoint_path, config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_peft_model.to(device)
# 在推理前，通常建议将模型设置为评估模式
val_peft_model.eval()


# ===================== 测试模型 =====================
with open("data/data_vl_test.json", "r") as f:
    test_dataset = json.load(f)

rouge = Rouge()

rouge_score_avg = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
levenshtein_avg = 0
total_levenshtein = 0
bleu_score_avg = 0
total_samples = len(test_dataset)
correct_samples  = 0
for item in test_dataset:
    input_image_prompt = item["conversations"][0]["value"]
    origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    origin_image_path = f"data/{origin_image_path}"
    true_output = item["conversations"][1]["value"]
    chinese_character = os.path.basename(origin_image_path).split('.')[0]
    print(f"Processing character: {chinese_character}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": origin_image_path},
                {"type": "text", "text": "请你用中文描述图片中汉字的笔画顺序"},
            ],
        }
    ]
    pred_output = predict(messages, val_peft_model)
    print(f"Predicted: {pred_output.strip()}")
    print(f"True     : {true_output.strip()}")
    if pred_output.strip() == true_output.strip():
        correct_samples += 1

    # 计算 BLEU 分数
    bleu_score = sentence_bleu([true_output.split()], pred_output.split())
    print(f"BLEU Score: {bleu_score}")
    bleu_score_avg += bleu_score

    # 计算 ROUGE 分数
    rouge_scores = rouge.get_scores(pred_output, true_output)[0]
    for key in rouge_score_avg:
        rouge_score_avg[key] += rouge_scores[key]["f"]

    # 计算 Levenshtein 距离
    levenshtein_avg += levenshtein_distance(pred_output.strip(), true_output.strip())

# 平均化指标
total_samples = max(total_samples, 1)  # 防止除以零
bleu_score_avg /= total_samples
for key in rouge_score_avg:
    rouge_score_avg[key] /= total_samples
levenshtein_avg /= total_samples

print(f"AVG BLEU Score: {bleu_score_avg}")
print(f"AVG ROUGE Scores: {rouge_score_avg}")
print(f"AVG Levenshtein Distance: {levenshtein_avg}")
accuracy = correct_samples / total_samples
print(f"Test Accuracy: {accuracy:.4f} ({correct_samples}/{total_samples})")