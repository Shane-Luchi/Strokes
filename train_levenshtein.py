import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from qwen_vl_utils import process_vision_info
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

# 加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.resize_token_embeddings(len(tokenizer))  # 🚨 调整Embedding以适配新token

# ===================== 处理函数 =====================
def process_func(example):
    MAX_LENGTH = 8192
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    file_path = f"data/{file_path}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": "请你用中文描述图片中汉字的笔画顺序"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}
    instruction = inputs

    # ======= 添加EOS Token到标签末尾 ========
    response = tokenizer(f"{output_content}{EOS_TOKEN}", add_special_tokens=False)
    #print("response input_ids:", response["input_ids"])
    input_ids = instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    vocab_size = tokenizer.vocab_size
    labels = [label if label == -100 or (0 <= label < vocab_size) or (label == tokenizer.eos_token_id) else -100 for label in labels]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs["pixel_values"] = torch.tensor(inputs["pixel_values"])
    inputs["image_grid_thw"] = torch.tensor(inputs["image_grid_thw"]).squeeze(0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
    }

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

# ===================== 自定义 Trainer =====================
class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        self.model.eval()
        total_samples = len(eval_dataset)
        correct_samples = 0
        bleu_score_avg = 0
        rouge = Rouge()
        rouge_score_avg = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
        levenshtein_avg = 0

        for example in eval_dataset:
            true_output = example["conversations"][1]["value"]
            input_content = example["conversations"][0]["value"]
            file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
            file_path = f"data/{file_path}"
            chinese_character = os.path.basename(file_path).split('.')[0]
            print(f"Processing character: {chinese_character}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"{file_path}",
                            "resized_height": 280,
                            "resized_width": 280,
                        },
                        {"type": "text", "text": "请你用中文描述图片中汉字的笔画顺序"},
                    ],
                }
            ]
            pred_output = predict(messages, self.model)

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
        print(f"Validation Accuracy: {accuracy:.4f} ({correct_samples}/{total_samples})")

        metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        metrics["eval_accuracy"] = accuracy
        metrics["eval_bleu"] = bleu_score_avg
        metrics["eval_rouge"] = rouge_score_avg
        metrics["eval_levenshtein"] = levenshtein_avg
        return metrics

        
# #####
# # ===================== 加载数据 =====================
# with open("data/train_data.json", "r") as f:
#     data = json.load(f)
# # 随机打乱数据
# random.seed(42)
# random.shuffle(data)


# total_size = len(data)
# train_size = int(total_size * 0.9)
# val_size = int(total_size * 0.05)
# test_size = total_size - train_size - val_size

# train_data = data[:train_size]
# val_data = data[train_size:train_size + val_size]
# test_data = data[train_size + val_size:]

# # extra_indices = [i for i in range(0, 8105, 100)]
# # extra_data = [data[i] for i in extra_indices]
# # # 将 extra_data 放在 val_data 的前面
# # val_data = extra_data + val_data



# with open("data/data_vl_train.json", "w") as f:
#     json.dump(train_data, f)
# with open("data/data_vl_val.json", "w") as f:
#     json.dump(val_data, f)
# with open("data/data_vl_test.json", "w") as f:
#     json.dump(test_data, f)


# train_ds = Dataset.from_json("data/data_vl_train.json")
# val_ds = Dataset.from_json("data/data_vl_val.json")

# train_dataset = train_ds.map(process_func, desc="处理后的数据集")
# val_dataset = val_ds.map(process_func, desc="处理后的验证数据集")

# train_dataset.save_to_disk("data/train_dataset_processed")
# val_dataset.save_to_disk("data/val_dataset_processed")
# #####

# 加载处理后数据
from datasets import load_from_disk
train_dataset = load_from_disk("data/train_dataset_processed")
val_dataset = load_from_disk("data/val_dataset_processed")


# ===================== 训练参数 =====================
args = TrainingArguments(
    output_dir="./output_levenshtein/Qwen2-VL-2B",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=50,
    save_steps=100,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=2000,
    save_on_each_node=True,
    gradient_checkpointing=False,
    report_to="none",
    ddp_find_unused_parameters=False,
    bf16=True,
    dataloader_num_workers=8,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=8,
)

# 不再使用 LoRA 配置，直接进行全量微调
trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

print("--------------starting to train---------------")
trainer.train()

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
    pred_output = predict(messages, model)
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


