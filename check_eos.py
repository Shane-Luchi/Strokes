from datasets import load_from_disk
import torch
import json
train_dataset = load_from_disk("data/train_dataset_processed")
val_dataset = load_from_disk("data/val_dataset_processed")
from modelscope import snapshot_download, AutoTokenizer
from qwen_vl_utils import process_vision_info

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)

tokenizer = AutoTokenizer.from_pretrained(
    "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/", use_fast=False, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("/home/LLMs/Qwen/Qwen2-VL-2B-Instruct")

# 添加 EOS Token 到 tokenizer
EOS_TOKEN = "<EOS>"
tokenizer.add_special_tokens({'eos_token': EOS_TOKEN})
EOS_TOKEN_ID = tokenizer.eos_token_id

vocab_size = tokenizer.vocab_size
print("vocab_size\n\n\n\n",vocab_size)

# for i in range(3):
#     sample = train_dataset[i]
#     print(f"样例 {i + 1}:")
#     print("input_ids:", sample["input_ids"])
#     print("attention_mask:", sample["attention_mask"])
#     print("labels:", sample["labels"])
#     pixel_values_tensor = torch.tensor(sample["pixel_values"])
#     print("pixel_values shape:", pixel_values_tensor.shape)
#     print("image_grid_thw:", sample["image_grid_thw"])
#     # 解码查看文本
#     print("Decoded input_ids:", tokenizer.decode(sample["input_ids"]))
#     print("Decoded labels:", tokenizer.decode([x for x in sample["labels"] if x != -100]))
#     print("-" * 50)

# from datasets import load_from_disk
# import torch

# train_dataset = load_from_disk("data/train_dataset_processed")
# sample = train_dataset[0]  # 取第一个样例


# # 检查每个 token
# for i, token_id in enumerate(sample["input_ids"]):
#     decoded = tokenizer.convert_ids_to_tokens([token_id])
#     if decoded[0] is None:
#         print(f"Invalid token at index {i}: {token_id}")
#     else:
#         print(f"Index {i}: {token_id} -> {decoded[0]}")
# #print(tokenizer.decode([101209]))  # 检查这个 token 是什么



# # 加载模型
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# )
# model.resize_token_embeddings(len(tokenizer))  # 🚨 调整Embedding以适配新token

# output_content = "横"  # 假设目标输出是“横”，你可以替换为实际数据中的值

# # 生成 response
# response = tokenizer(f"{output_content}{EOS_TOKEN}", add_special_tokens=False)
# print("Response input_ids:", response["input_ids"])
# print("Response text:", tokenizer.decode(response["input_ids"], skip_special_tokens=False))
# print("EOS_TOKEN_ID:", tokenizer.eos_token_id)


# def predict(messages, model, max_new_tokens=64, temperature=1.0, top_p=0.9, top_k=50):
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")

#     generated_ids = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         eos_token_id=EOS_TOKEN_ID,    # 指定终止符号
#         pad_token_id=tokenizer.pad_token_id,
#         temperature=temperature,      # 控制生成的随机性，多样性
#         top_p=top_p,                  # nucleus sampling
#         top_k=top_k                   # 限制采样范围
#     )

#     generated_ids_trimmed = [
#         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False,
#     )
#     return output_text[0]

sample = train_dataset[0]
print("Sample type:", type(sample))
print("Sample keys:", sample.keys())
print("labels type:", type(sample["labels"]))
print("labels value:", sample["labels"])

# for sample in train_dataset[0]:
#     labels = sample["labels"]
#     print("labels type:", type(labels))
#     print("labels value:", labels)
labels = sample["labels"]   
# 根据类型处理
if isinstance(labels, str):
    import ast
    labels = ast.literal_eval(labels)  # 将字符串转换为列表
elif isinstance(labels, torch.Tensor):
    labels = labels.tolist()  # 将张量转换为列表

# 解码
non_ignored_labels = [x for x in labels if x != -100]
print("non_ignored_labels:", non_ignored_labels)
print("decoded labels:", tokenizer.decode(non_ignored_labels, skip_special_tokens=False))

# # ===================== 测试模型 =====================
# with open("data/data_vl_test.json", "r") as f:
#     test_dataset = json.load(f)

# for item in test_dataset:
#     input_image_prompt = item["conversations"][0]["value"]
#     origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
#     origin_image_path = f"data/{origin_image_path}"
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": origin_image_path},
#                 {"type": "text", "text": "请你用中文描述图片中汉字的笔画顺序"},
#             ],
#         }
#     ]
#     response = predict(messages, model)  # 使用全量微调后的模型进行预测

#     # response = tokenizer(f"{output_content}{EOS_TOKEN}", add_special_tokens=False)
#     print("Response input_ids:", response["input_ids"])
#     print("Response text:", tokenizer.decode(response["input_ids"], skip_special_tokens=False))
#     print("EOS_TOKEN_ID:", tokenizer.eos_token_id)
