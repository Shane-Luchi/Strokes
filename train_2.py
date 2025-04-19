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
from nltk.translate.bleu_score import sentence_bleu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    vocab_size = tokenizer.vocab_size
    labels = [label if label == -100 or (0 <= label < vocab_size) else -100 for label in labels]

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

def predict(messages, model):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]

# 自定义 Trainer 来计算验证集正确率
class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        # 切换到评估模式
        self.model.eval()
        total_samples = len(eval_dataset)
        correct_samples = 0

        for example in eval_dataset:
            # 获取真实的输出
            true_output = example["conversations"][1]["value"]
            # 构造输入消息
            input_content = example["conversations"][0]["value"]
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
            # 预测输出
            pred_output = predict(messages, self.model)
            # 打印预测输出和真实值
            print(f"Predicted: {pred_output.strip()}")
            print(f"True     : {true_output.strip()}")
            # 判断是否正确（这里假设完全匹配才算正确）
            if pred_output.strip() == true_output.strip():
                correct_samples += 1

            bleu_score = sentence_bleu([true_output.split()], pred_output.split())
            print(f"BLEU Score: {bleu_score}")

        accuracy = correct_samples / total_samples if total_samples > 0 else 0
        print(f"Validation Accuracy: {accuracy:.4f} ({correct_samples}/{total_samples})")
        
        # 调用父类的 evaluate 方法以保持原有功能
        metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        metrics["eval_accuracy"] = accuracy
        return metrics

    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        # 重写 train 方法，在每个 epoch 结束时进行评估
        result = super().train(resume_from_checkpoint, trial, **kwargs)
        return result

# 加载模型和处理器
tokenizer = AutoTokenizer.from_pretrained(
    "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/", use_fast=False, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("/home/LLMs/Qwen/Qwen2-VL-2B-Instruct")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.enable_input_require_grads()

# 处理数据集
train_json_path = "data/train_data.json"
with open(train_json_path, "r") as f:
    data = json.load(f)

total_size = len(data)
train_size = int(total_size * 0.95)
val_size = int(total_size * 0.04)
test_size = total_size - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

extra_indices = [i for i in range(0, 8105, 100)]
extra_data = [data[i] for i in extra_indices]
val_data.extend(extra_data)  # 在 val_data 上追加

with open("data/data_vl_train.json", "w") as f:
    json.dump(train_data, f)
with open("data/data_vl_val.json", "w") as f:
    json.dump(val_data, f)
with open("data/data_vl_test.json", "w") as f:
    json.dump(test_data, f)

#####
# train_ds = Dataset.from_json("data/data_vl_train.json")
# val_ds = Dataset.from_json("data/data_vl_val.json")

# train_dataset = train_ds.map(process_func, desc="处理后的数据集")
# val_dataset = val_ds.map(process_func, desc="处理后的验证数据集")

# train_dataset.save_to_disk("data/train_dataset_processed")
# val_dataset.save_to_disk("data/val_dataset_processed")
# ####

from datasets import load_from_disk
train_dataset = load_from_disk("data/train_dataset_processed")
val_dataset = load_from_disk("data/val_dataset_processed")

args = TrainingArguments(
    output_dir="./output3/Qwen2-VL-2B",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # 增加梯度累积
    logging_steps=10,
    num_train_epochs=50,  # 增加训练轮数
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
    per_device_eval_batch_size=16,
)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    inference_mode=False,
    r=32,  # 减小 r
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)
peft_model = get_peft_model(model, config)



# 配置 CustomTrainer
trainer = CustomTrainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)





# 训练模型
print("--------------starting to train---------------")
trainer.train()

# ===测试模式===
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=True,
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)





print("--------------starting to test---------------")
val_peft_model = PeftModel.from_pretrained(
    model, model_id="./output3/Qwen2-VL-2B/checkpoint-12000", config=val_config
)

with open("data/data_vl_test.json", "r") as f:
    test_dataset = json.load(f)

test_image_list = []
for item in test_dataset:
    input_image_prompt = item["conversations"][0]["value"]
    origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    origin_image_path = f"data/{origin_image_path}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": origin_image_path},
                {"type": "text", "text": "请你用中文描述图片中汉字的笔画顺序"},
            ],
        }
    ]
    response = predict(messages, val_peft_model)
    messages.append({"role": "assistant", "content": f"{response}"})
    print(messages[-1])