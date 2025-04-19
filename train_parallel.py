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
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# Setup logging with TensorBoard
summarywriter = SummaryWriter(log_dir="./logs")
import torch.distributed as dist

# 初始化分布式进程组
dist.init_process_group(backend='nccl', init_method='env://')

# 获取当前进程的 local_rank 和设备
local_rank = int(os.environ['LOCAL_RANK'])
print("local_rank:", local_rank)
torch.cuda.set_device(local_rank)

# 设置环境变量，指定GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # 使用 GPU 2 和 GPU 3

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

# 将模型移动到当前设备（GPU）
model = model.to(local_rank)

model = DDP(model, device_ids=[local_rank])  # 假设使用两个GPU
# 确保所有进程同步
dist.barrier()

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
    def __init__(self, *args, summarywriter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.summarywriter = summarywriter  # 将tb_writer传递给Trainer类
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        self.model.eval()
        total_samples = len(eval_dataset)
        correct_samples = 0

        for example in eval_dataset:
            true_output = example["conversations"][1]["value"]
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
            pred_output = predict(messages, self.model)
            print(f"Predicted: {pred_output.strip()}")
            print(f"True     : {true_output.strip()}")
            if pred_output.strip() == true_output.strip():
                correct_samples += 1
            bleu_score = sentence_bleu([true_output.split()], pred_output.split())
            print(f"BLEU Score: {bleu_score}")

        accuracy = correct_samples / total_samples if total_samples > 0 else 0
        print(f"Validation Accuracy: {accuracy:.4f} ({correct_samples}/{total_samples})")
        metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        metrics["eval_accuracy"] = accuracy

        # 记录评估结果到 TensorBoard
        summarywriter.add_scalar('eval/accuracy', accuracy, self.state.global_step)

        return metrics

    def training_step(self, model, inputs):
        # 执行训练步骤
        loss = super().training_step(model, inputs)

        # 获取当前epoch的loss，记录到TensorBoard
        summarywriter.add_scalar('train/loss', loss.item(), self.state.global_step)
        
        return loss

# # ===================== 加载数据 =====================
# with open("data/train_data.json", "r") as f:
#     data = json.load(f)

# total_size = len(data)
# train_size = int(total_size * 0.97)
# val_size = int(total_size * 0.02)
# test_size = total_size - train_size - val_size

# train_data = data[:train_size]
# val_data = data[train_size:train_size + val_size]
# test_data = data[train_size + val_size:]

# extra_indices = [i for i in range(0, 8105, 100)]
# extra_data = [data[i] for i in extra_indices]
# # 将 extra_data 放在 val_data 的前面
# val_data = extra_data + val_data

# with open("data/data_vl_train.json", "w") as f:
#     json.dump(train_data, f)
# with open("data/data_vl_val.json", "w") as f:
#     json.dump(val_data, f)
# with open("data/data_vl_test.json", "w") as f:
#     json.dump(test_data, f)

# 加载处理后数据
from datasets import load_from_disk
train_dataset = load_from_disk("data/train_dataset_processed")
val_dataset = load_from_disk("data/val_dataset_processed")

# ===================== 训练参数 =====================
args = TrainingArguments(
    output_dir="./output3/Qwen2-VL-2B",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
    logging_dir="./logs",
    num_train_epochs=50,
    save_steps=100,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=2000,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="tensorboard",
    ddp_find_unused_parameters=False,  # 设置分布式数据并行
    bf16=True,
    dataloader_num_workers=8,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=4,
    local_rank=-1,  # 用于分布式训练
)

# 不再使用 LoRA 配置，直接进行全量微调
trainer = CustomTrainer(
    model=model,
    args=args,
    summarywriter = summarywriter,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

print("--------------starting to train---------------")
trainer.train()

# ===================== 测试模型 =====================
with open("data/data_vl_test.json", "r") as f:
    test_dataset = json.load(f)

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
    response = predict(messages, model)  # 使用全量微调后的模型进行预测
    messages.append({"role": "assistant", "content": f"{response}"})
    print(messages[-1])