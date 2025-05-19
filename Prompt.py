"""
Comment out line 116

run in bash:

python Prompt.py

or run in bash:

nohup ~/anaconda3/envs/Strokes/bin/python Prompt.py > log/Prompt_Qwen_finetune_516.log 2>&1 &

origin Qwen2-VL-2B-Instruct
nohup ~/anaconda3/envs/Strokes/bin/python Prompt.py > log/Prompt_Qwen_finetune_516.log 2>&1 &

"""

import json
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from transformers import CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from Levenshtein import distance as levenshtein_distance
import pandas as pd
import requests
from io import BytesIO
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 设置 CUDA 设备（根据你的硬件调整）
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 初始化 ROUGE 和结果列表
rouge = Rouge()
results = []

# 加载测试集，与 train_levenshtein.py 一致
with open("data/data_vl_test.json", "r", encoding="utf-8") as f:
    test_dataset = json.load(f)
print(f"测试集大小: {len(test_dataset)}")


# 评估函数
def evaluate_model(model_name, predict_fn, test_dataset):
    total_samples = len(test_dataset)
    correct_samples = 0
    bleu_score_avg = 0
    rouge_score_avg = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
    levenshtein_avg = 0

    for example in test_dataset:
        true_output = example["conversations"][1]["value"].strip()
        input_content = example["conversations"][0]["value"]
        file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[
            0
        ]
        file_path = f"data/{file_path}"
        chinese_character = os.path.basename(file_path).split(".")[0]
        print(f"处理字符: {chinese_character}，模型: {model_name}")

        # 生成预测
        pred_output = predict_fn(file_path).strip()
        pred_output = pred_output.replace("、", " ")
        print(f"预测: {pred_output}")
        print(f"真实: {true_output}")

        # 计算准确率
        if pred_output == true_output:
            correct_samples += 1
        pred_output = pred_output.replace("、", " ")
        # 计算 BLEU 分数
        bleu_score = sentence_bleu([true_output.split()], pred_output.split())
        bleu_score_avg += bleu_score
        print(f"BLEU 分数: {bleu_score}")

        # 计算 ROUGE 分数
        rouge_scores = rouge.get_scores(pred_output, true_output)[0]
        for key in rouge_score_avg:
            rouge_score_avg[key] += rouge_scores[key]["f"]
        print(f"Rouge 分数：{rouge_scores}")

        # 计算 Levenshtein 距离
        levenshtein_dist = levenshtein_distance(
            pred_output.split(), true_output.split()
        )
        levenshtein_avg += levenshtein_dist
        print(f"Levenshtein 距离: {levenshtein_dist}")

    # 平均指标
    bleu_score_avg /= total_samples
    for key in rouge_score_avg:
        rouge_score_avg[key] /= total_samples
    levenshtein_avg /= total_samples
    accuracy = correct_samples / total_samples if total_samples > 0 else 0

    return {
        "模型": model_name,
        "准确率": accuracy,
        "BLEU": bleu_score_avg,
        "ROUGE-1": rouge_score_avg["rouge-1"],
        "ROUGE-2": rouge_score_avg["rouge-2"],
        "ROUGE-L": rouge_score_avg["rouge-l"],
        "Levenshtein": levenshtein_avg,
        "正确样本数": correct_samples,
        "总样本数": total_samples,
    }


# 1. Qwen2-VL-2B-Instruct
def predict_qwen2vl(image_path):
    model_path = "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path)
    # model_path = "/home/zsy/GithubCode/Strokes/output4/Qwen2-VL-2B/checkpoint-7000"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {
                    "type": "text",
                    "text": "请你用中文描述图片中汉字的笔画顺序，直接给出笔画顺序，从下面五种笔画的范围内选取：横、竖、撇、捺、横折钩。每个笔画之间用空格空开，不要采用顿号，逗号等其他符号。",
                },
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, padding=True, return_tensors="pt"
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )[0]
    return output_text


# 2. LLaVA-13B
def predict_llava(image_path):
    model_path = "/home/LLMs/llava/llava-1.5-7b-hf"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name
    )
    model.to("cuda")

    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor(image, return_tensors="pt")["pixel_values"].to(
        "cuda"
    )

    prompt = "请你用中文描述图片中汉字的笔画顺序"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs.input_ids,
        pixel_values=image_tensor,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.9,
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text


# 3. CLIP + GPT-2
def predict_clip_gpt2(image_path):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(
        "cuda"
    )
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")

    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to("cuda")
    image_features = clip_model.get_image_features(**inputs)

    prompt = "请你用中文描述图片中汉字的笔画顺序"
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = gpt2_model.generate(
        input_ids, max_length=64, do_sample=True, temperature=0.9
    )
    output_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text


# 4. Grok 3（占位符，需替换为实际 API 或本地推理）
def predict_grok3(image_path):
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        # 假设 xAI API 端点（请替换为真实端点和密钥）
        response = requests.post(
            "https://api.x.ai/grok3/predict",
            files={"image": (image_path, image_data)},
            data={"prompt": "请你用中文描述图片中汉字的笔画顺序"},
            headers={"Authorization": "Bearer YOUR_API_KEY"},
        )
        return response.json().get("text", "错误: 无响应")
    except Exception as e:
        return f"错误: {str(e)}"


# 评估所有模型
models = [
    ("Qwen2-VL-2B-Instruct", predict_qwen2vl),
    # ("LLaVA-7B", predict_llava)
    # ("CLIP+GPT2", predict_clip_gpt2),
    # ("Grok3", predict_grok3)
]

for model_name, predict_fn in models:
    print(f"\n评估模型: {model_name}...")
    result = evaluate_model(model_name, predict_fn, test_dataset)
    results.append(result)
    print(result)

# # 保存结果为 CSV
# results_df = pd.DataFrame(results)
# results_df.to_csv("model_comparison_results.csv", index=False, encoding="utf-8-sig")
# print("\n结果已保存至 model_comparison_results.csv")
# print(results_df)
