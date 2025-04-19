import torch
from modelscope import AutoTokenizer
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import json
import os
from nltk.translate.bleu_score import sentence_bleu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 加载基础模型和处理器
tokenizer = AutoTokenizer.from_pretrained(
    "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/", use_fast=False, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("/home/LLMs/Qwen/Qwen2-VL-2B-Instruct")

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")

# 定义推理函数
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

    # 生成输出，减少随机性以突出模型差异
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.1,  # 降低温度，减少随机性
        top_p=1.0,  # 禁用 top-p 采样，使用贪婪解码
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

# 配置 LoRA 用于推理
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    inference_mode=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

# 加载训练好的 LoRA 模型
model_path = "./output2/Qwen2-VL-2B/checkpoint-12000"
val_peft_model = PeftModel.from_pretrained(
    base_model, model_id=model_path, config=val_config
).to("cuda")

# 调试：确认 LoRA 适配器是否加载并启用
print("Active adapters:", val_peft_model.active_adapters)  # 修正：移除括号，直接访问属性
# 修正：get_adapter_state 也可能是属性，视 peft 版本而定
try:
    print("Adapter layers enabled:", val_peft_model.get_adapter_state)
except AttributeError:
    print("get_adapter_state not available in this peft version.")

# 调试：检查 LoRA 权重是否非零
for name, param in val_peft_model.named_parameters():
    if "lora" in name:
        print(f"{name}: norm = {param.data.norm()}")  # 权重范数应非零

# 读取测试数据
test_json_path = "data/data_vl_test2.json"
with open(test_json_path, "r") as f:
    test_dataset = json.load(f)

# 进行推理
print("--------------starting inference---------------")
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
    # 基座模型推理
    base_response = predict(messages, base_model)
    print(f"基模型预测: {base_response}")
    
    # 微调模型推理
    peft_response = predict(messages, val_peft_model)
    print(f"图像路径: {origin_image_path}")
    print(f"微调模型预测: {peft_response}")
    print("------------------------")