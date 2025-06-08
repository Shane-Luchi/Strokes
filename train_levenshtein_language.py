"""
CUDA_VISIBLE_DEVICES=0,1 python train_levenshtein_language.py
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_levenshtein_language.py
CUDA_VISIBLE_DEVICES=3 nohup python train_levenshtein_language.py > logs/train_levenshtein_language.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_levenshtein_language.py
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 train_levenshtein_language.py
禁止采用P2P通信

CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 train_levenshtein_language.py

CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 nohup torchrun --nproc_per_node=2 train_levenshtein_language.py > logs/train_levenshtein_language_qwen2.5_7B_$(date +%Y%m%d_%H%M%S).log 2>&1 &

因为7B会OOM，所以采用了accelerate来多卡训练加速，速度不知道，但是运行起来了，多卡推理有问题，重复推理。
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 accelerate launch train_levenshtein_language.py
"""

from swanlab.integration.transformers import SwanLabCallback
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
)

import json
import time
import os
import random
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from Levenshtein import distance as levenshtein_distance

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SWANLAB_PROJECT"] = "qwen2.5-7B-Instruct"
os.environ["SWANLAB_WORKSPACE"] = "Shane-Luchi"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# ====================== 新增EOS定义 ======================
EOS_TOKEN = "<EOS>"
MODEL_PATH = "/home/LLMs/Qwen/Qwen2.5-14B-Instruct"
MODEL_NAME = "Qwen2.5-14B-Instruct"
custom_experiment_name = f"train_language_{MODEL_NAME}_{time.strftime('%Y%m%d-%H%M%S')}"

# 加载 tokenizer 和 processor
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=False, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# 添加 EOS Token 到 tokenizer
tokenizer.add_special_tokens({"eos_token": EOS_TOKEN})
EOS_TOKEN_ID = tokenizer.eos_token_id

# 加载模型
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# )
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
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
    messages = [
        {
            "role": "user",
            "content": f"""
                请严格按照以下五种基本笔画类型（横、竖、撇、捺、横折钩）分解汉字「{input_content}」的笔顺，并按书写顺序逐笔列出。  
                要求：  
                1. 只使用这五种笔画名称，不细分（如“点”归为“捺”，“竖钩”归为“竖”等）。
                2. 按正确笔顺逐笔输出，输出结果放在一行中，每个笔画之间用一个空格隔开，格式示例：横 竖 撇 捺 横折钩  

                例如「木」字的笔顺：横 竖 撇 捺  

                现在请分析「{input_content}」：
                """,
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}
    instruction = inputs

    # ======= 添加EOS Token到标签末尾 ========
    response = tokenizer(f"{output_content}{EOS_TOKEN}", add_special_tokens=False)
    # print("response input_ids:", response["input_ids"])
    input_ids = (
        instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"][0])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    vocab_size = tokenizer.vocab_size
    labels = [
        (
            label
            if label == -100
            or (0 <= label < vocab_size)
            or (label == tokenizer.eos_token_id)
            else -100
        )
        for label in labels
    ]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ===================== 推理函数 =====================
def predict(messages, model, max_new_tokens=64, temperature=1.0, top_p=0.9, top_k=50):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=EOS_TOKEN_ID,  # 指定终止符号
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature,  # 控制生成的随机性，多样性
        top_p=top_p,  # nucleus sampling
        top_k=top_k,  # 限制采样范围
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
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
        counter = 0
        for example in eval_dataset:
            true_output = example["conversations"][1]["value"]
            input_content = example["conversations"][0]["value"]
            counter += 1
            print(
                f"Processing character: {input_content}  id: {example['id']} counter: {counter}/{total_samples}"
            )

            messages = [
                {
                    "role": "user",
                    "content": f"""
                请严格按照以下五种基本笔画类型（横、竖、撇、捺、横折钩）分解汉字「{input_content}」的笔顺，并按书写顺序逐笔列出。  
                要求：  
                1. 只使用这五种笔画名称，不细分（如“点”归为“捺”，“竖钩”归为“竖”等）。
                2. 按正确笔顺逐笔输出，输出结果放在一行中，每个笔画之间用一个空格隔开，格式示例：横 竖 撇 捺 横折钩  

                例如「木」字的笔顺：横 竖 撇 捺  

                现在请分析「{input_content}」：
                """,
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
            levenshtein_avg += levenshtein_distance(
                pred_output.strip(), true_output.strip()
            )
            print(
                "Levenshtein Distance: ",
                levenshtein_distance(pred_output.strip(), true_output.strip()),
            )
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
        print(
            f"Validation Accuracy: {accuracy:.4f} ({correct_samples}/{total_samples})"
        )

        metrics = super().evaluate(
            eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        metrics["eval_accuracy"] = accuracy
        metrics["eval_bleu"] = bleu_score_avg
        metrics["eval_rouge"] = rouge_score_avg
        metrics["eval_levenshtein"] = levenshtein_avg
        return metrics


# #####
# # ===================== 加载数据 =====================
# with open("data/train_data_language.json", "r") as f:
#     data = json.load(f)
# # 随机打乱数据
# random.seed(42)
# random.shuffle(data)


# total_size = len(data)
# train_size = int(total_size * 0.9)
# val_size = int(total_size * 0.05)
# test_size = total_size - train_size - val_size

# train_data = data[:train_size]
# val_data = data[train_size : train_size + val_size]
# test_data = data[train_size + val_size :]

# # extra_indices = [i for i in range(0, 8105, 100)]
# # extra_data = [data[i] for i in extra_indices]
# # # 将 extra_data 放在 val_data 的前面
# # val_data = extra_data + val_data


# with open(f"data/data_vl_train_language_{MODEL_NAME}.json", "w") as f:
#     json.dump(train_data, f)
# with open(f"data/data_vl_val_language_{MODEL_NAME}.json", "w") as f:
#     json.dump(val_data, f)
# with open(f"data/data_vl_test_language_{MODEL_NAME}.json", "w") as f:
#     json.dump(test_data, f)


# train_ds = Dataset.from_json(f"data/data_vl_train_language_{MODEL_NAME}.json")
# val_ds = Dataset.from_json(f"data/data_vl_val_language_{MODEL_NAME}.json")

# train_dataset = train_ds.map(process_func, desc="处理后的数据集")
# val_dataset = val_ds.map(process_func, desc="处理后的验证数据集")

# train_dataset.save_to_disk(f"data/train_dataset_processed_language_{MODEL_NAME}")
# val_dataset.save_to_disk(f"data/val_dataset_processed_language_{MODEL_NAME}")
# #####

# 加载处理后数据
from datasets import load_from_disk

train_dataset = load_from_disk(f"data/train_dataset_processed_language_{MODEL_NAME}")
val_dataset = load_from_disk(f"data/val_dataset_processed_language_{MODEL_NAME}")


# ===================== 训练参数 =====================
args = TrainingArguments(
    output_dir=f"./output_levenshtein_language_{MODEL_NAME}/{MODEL_NAME}",
    optim="adamw_bnb_8bit",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=20,
    save_steps=100,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=1000,
    save_on_each_node=True,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    bf16=True,
    dataloader_num_workers=8,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=8,
    report_to="none",
    # report_to="swanlab",
    # report_to_swanlab=True,
    # run_name="train_levenshtein_language_522",
)
# with open("data/data_vl_val_language.json", "r") as f:
#     val_dataset = json.load(f)
# 不再使用 LoRA 配置，直接进行全量微调
swanlab_callback = SwanLabCallback(
    project=os.environ.get("SWANLAB_PROJECT"),  # 使用环境变量中定义的项目名
    workspace=os.environ.get("SWANLAB_WORKSPACE"),  # 使用环境变量中定义的工作空间名
    experiment_name=custom_experiment_name,
    description=f"Fine-tuning {MODEL_NAME} for Levenshtein language task (stroke prediction).",
)

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

print("--------------starting to train---------------")
trainer.train()

# ===================== 测试模型 =====================
with open(f"data/data_vl_test_language_{MODEL_NAME}.json", "r") as f:
    test_dataset = json.load(f)

rouge = Rouge()

rouge_score_avg = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
levenshtein_avg = 0
total_levenshtein = 0
bleu_score_avg = 0
total_samples = len(test_dataset)
correct_samples = 0
counter = 0
for item in test_dataset:
    input_image_prompt = item["conversations"][0]["value"]
    true_output = item["conversations"][1]["value"]
    counter += 1
    print(
        f"Processing character: {input_image_prompt}  id: {item['id']}  counter: {counter}/{total_samples}"
    )

    messages = [
        {
            "role": "user",
            "content": f"""
                请严格按照以下五种基本笔画类型（横、竖、撇、捺、横折钩）分解汉字「{input_image_prompt}」的笔顺，并按书写顺序逐笔列出。  
                要求：  
                1. 只使用这五种笔画名称，不细分（如“点”归为“捺”，“竖钩”归为“竖”等）。
                2. 按正确笔顺逐笔输出，输出结果放在一行中，每个笔画之间用一个空格隔开，格式示例：横 竖 撇 捺 横折钩  

                例如「木」字的笔顺：横 竖 撇 捺  

                现在请分析「{input_image_prompt}」：
                """,
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
    print(
        "Levenshtein Distance: ",
        levenshtein_distance(pred_output.strip(), true_output.strip()),
    )
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
