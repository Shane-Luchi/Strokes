import torch
from datasets import load_dataset
from PIL import Image as PILImage
from modelscope import AutoTokenizer
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    AutoConfig
)
from accelerate import Accelerator
import json
import os
import re
import numpy as np
from typing import List, Dict, Any, Optional
import io
import ast
import logging
from datetime import datetime
import torch

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mmmu_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 配置参数 ---
MODEL_BASE_PATH = "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/"
CHECKPOINT_PATH = "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/" 
MMMU_DATASET_PATH = "MMMU/MMMU"
MMMU_SPLIT = "validation"
MMMU_CUSTOM_CACHE_DIR = "data/MMMU_hf_cache"
os.makedirs(MMMU_CUSTOM_CACHE_DIR, exist_ok=True)

# --- 科目定义 ---
ALL_MMMU_AVAILABLE_SUBJECTS = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
    'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
    'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage',
    'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy',
    'Physics', 'Psychology', 'Public_Health', 'Sociology'
]

MMMU_SUBJECTS_TO_EVALUATE = ALL_MMMU_AVAILABLE_SUBJECTS

# --- 模型参数 ---
MAX_NEW_TOKENS_MMMU = 20
TEMPERATURE_MMMU = 0.1
TOP_P_MMMU = 0.9
TOP_K_MMMU = 40
EOS_TOKEN_STRING = "<EOS>"

def parse_options(options: Any) -> List[str]:
    """统一解析不同格式的选项输入"""
    if isinstance(options, list):
        return [str(opt) for opt in options]
    
    try:
        parsed = ast.literal_eval(options)
        if isinstance(parsed, list):
            return [str(opt) for opt in parsed]
    except (ValueError, SyntaxError):
        pass
    
    return [opt.strip() for opt in options.split('\n') if opt.strip()]

def load_mmmu_dataset(subject: str) -> torch.utils.data.DataLoader:
    """加载指定科目的数据集"""
    dataset = load_dataset(
        MMMU_DATASET_PATH,
        name=subject,
        split=MMMU_SPLIT,
        trust_remote_code=True,
        cache_dir=MMMU_CUSTOM_CACHE_DIR
    )
    
    def collate_fn(batch):
        return batch[0] if batch else None
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4
    )

def predict_for_mmmu(
    question_text: str,
    options: Any,
    images: List[PILImage.Image],
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer
) -> str:
    """执行模型推理"""
    parsed_options = parse_options(options)
    if not parsed_options:
        return "Error: No valid options"
    
    # 构建prompt
    prompt_lines = [
        "The following is a multiple-choice question. Please choose the correct option:",
        f"\nQuestion: {question_text}\nOptions:"
    ]
    for i, opt in enumerate(parsed_options):
        label = chr(ord('A') + i)
        prompt_lines.append(f"{label}) {opt}")
    prompt_lines.append("\nAnswer:")
    prompt = "\n".join(prompt_lines)
    
    # 准备多模态输入
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    
    try:
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        vision_input, _ = process_vision_info(messages)
        
        inputs = processor(
            text=[text_input],
            images=vision_input,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_MMMU,
                temperature=TEMPERATURE_MMMU,
                top_p=TOP_P_MMMU,
                top_k=TOP_K_MMMU,
                do_sample=True
            )
        
        return processor.decode(outputs[0], skip_special_tokens=True).strip()
    
    except Exception as e:
        logger.error(f"推理错误: {str(e)}")
        return f"Error: {str(e)}"

def evaluate_subject(
    subject: str,
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    """评估单个科目"""
    results = {
        "accuracy": 0.0,
        "correct": 0,
        "total": 0,
        "errors": []
    }
    
    try:
        dataloader = load_mmmu_dataset(subject)
        total = len(dataloader.dataset)
        logger.info(f"开始评估 {subject} (共 {total} 题)")
        
        for batch_idx, example in enumerate(dataloader):
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"进度: {batch_idx+1}/{total} ({(batch_idx+1)/total:.1%})")
            
            # 准备数据
            qid = example.get("id", f"{subject}_{batch_idx}")
            images = []
            for i in range(1, 8):
                img_field = f"image_{i}"
                if img_field in example and example[img_field]:
                    try:
                        img = example[img_field]
                        if isinstance(img, PILImage.Image):
                            images.append(img.convert("RGB"))
                        elif isinstance(img, dict) and 'bytes' in img:
                            images.append(PILImage.open(io.BytesIO(img['bytes'])).convert("RGB"))
                    except Exception as e:
                        logger.warning(f"图像加载失败 {qid}: {str(e)}")
            
            # 跳过无效问题
            if not images and any(f"image_{i}" in example for i in range(1, 8)):
                results["errors"].append(f"{qid}: 无有效图像")
                continue
                
            # 推理
            output = predict_for_mmmu(
                example["question"],
                example["options"],
                images,
                model,
                processor,
                tokenizer
            )
            
            # 评估结果
            correct = str(example["answer"]).strip().upper()
            predicted = parse_mmmu_model_answer(output, len(parse_options(example["options"])))
            
            results["total"] += 1
            if predicted == correct:
                results["correct"] += 1
            
            torch.cuda.empty_cache()
        
        # 计算准确率
        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"] * 100
            
        logger.info(f"{subject} 评估完成: 准确率 {results['accuracy']:.2f}%")
        
    except Exception as e:
        logger.error(f"评估 {subject} 时出错: {str(e)}")
        results["errors"].append(str(e))
    
    return results

def main():
    # --- 初始化 ---
    accelerator = Accelerator(mixed_precision="bf16")
    
    # 加载tokenizer和processor
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_BASE_PATH,
        use_fast=False,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_BASE_PATH,
        trust_remote_code=True
    )
    
    # 处理特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 准备分布式训练
    model, processor = accelerator.prepare(model, processor)
    model.eval()
    
    # --- 科目分配 ---
    if accelerator.num_processes > 1:
        subjects = np.array_split(MMMU_SUBJECTS_TO_EVALUATE, accelerator.num_processes)[accelerator.process_index]
    else:
        subjects = MMMU_SUBJECTS_TO_EVALUATE
    
    logger.info(f"进程 {accelerator.process_index} 分配到的科目: {subjects}")
    
    # --- 执行评估 ---
    all_results = {}
    for subject in subjects:
        result = evaluate_subject(subject, model, processor, tokenizer)
        all_results[subject] = result
    
    # --- 结果收集和保存 ---
    if accelerator.is_main_process:
        # 等待所有进程完成
        accelerator.wait_for_everyone()
        
        # 收集所有结果
        if accelerator.num_processes > 1:
            for rank in range(1, accelerator.num_processes):
                rank_results = accelerator.gather_for_metrics(all_results)
                all_results.update(rank_results)
        
        # 计算总体指标
        total_correct = sum(r["correct"] for r in all_results.values())
        total_questions = sum(r["total"] for r in all_results.values())
        avg_accuracy = total_correct / total_questions * 100 if total_questions > 0 else 0.0
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"mmmu_results_{timestamp}.json"
        
        results = {
            "config": {
                "model": CHECKPOINT_PATH,
                "subjects": MMMU_SUBJECTS_TO_EVALUATE,
                "date": timestamp
            },
            "metrics": {
                "average_accuracy": avg_accuracy,
                "total_correct": total_correct,
                "total_questions": total_questions
            },
            "details": all_results
        }
        
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"评估完成! 平均准确率: {avg_accuracy:.2f}%")
        logger.info(f"结果已保存到 {result_file}")
    
    else:
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()