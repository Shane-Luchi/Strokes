'''

CUDA_VISIBLE_DEVICES=1 python evaluate_docvqatest_qwen_finetune.py 

CUDA_VISIBLE_DEVICES=1 nohup python evaluate_docvqatest_qwen_finetune.py  > evaluate_docvqatest_qwen_finetune_522.log 2>&1 &

line337:
        results_filename = f"mmmu_val_eval_subjects_ckpt_1100.json"

'''
import torch
from datasets import load_dataset, Image as HFImage
from PIL import Image as PILImage
from modelscope import AutoTokenizer
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    AutoConfig
)
import json
import os
# import re # re is not used if normalize_text doesn't use it.
from typing import List # Dict, Any are not strictly typed in a way that requires them here.
import io
import Levenshtein

# --- 配置参数 ---
# !!! 用户需要修改以下路径 !!!
MODEL_BASE_PATH = "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/"
CHECKPOINT_PATH = "./output_levenshtein/Qwen2-VL-2B/checkpoint-1000"
# CHECKPOINT_PATH = "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/" # 评估基础模型时使用

DOCVQA_DATASET_NAME = "lmms-lab/DocVQA"
DOCVQA_CONFIG_NAME = "DocVQA" # "DocVQA" as per your previous log, adjust if it's the default config (then use None)
DOCVQA_SPLIT = "validation"

HF_CACHE_DIR = "data/docvqa_hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
print(f"Hugging Face 数据集将使用缓存目录: {os.path.abspath(HF_CACHE_DIR)}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MAX_NEW_TOKENS_DOCVQA = 100
TEMPERATURE_DOCVQA = 0.0
TOP_P_DOCVQA = 0.9
TOP_K_DOCVQA = 40

EOS_TOKEN_STRING = "<EOS>" # As per your provided script

# --- 1. 加载 Tokenizer 和 Processor ---
print(f"从基础模型路径加载 Tokenizer: {MODEL_BASE_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_BASE_PATH,
    use_fast=False,
    trust_remote_code=True
)

print(f"从基础模型路径加载 Processor: {MODEL_BASE_PATH}")
processor = AutoProcessor.from_pretrained(
    MODEL_BASE_PATH,
    trust_remote_code=True
    # use_fast=True # Consider explicitly setting for future transformers versions
)

# --- 2. 处理 EOS 和 Pad Token ---
if tokenizer.eos_token:
    EOS_TOKEN_STRING = tokenizer.eos_token
    print(f"Using EOS token from tokenizer: '{EOS_TOKEN_STRING}'")
else:
    num_added_eos = tokenizer.add_special_tokens({'eos_token': EOS_TOKEN_STRING})
    if num_added_eos > 0:
        print(f"Added '{EOS_TOKEN_STRING}' as eos_token.")
    else:
        print(f"'{EOS_TOKEN_STRING}' already configured as eos_token.")
EOS_TOKEN_ID = tokenizer.eos_token_id

if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
    print(f"Tokenizer pad_token未设置，将设置为与eos_token相同。")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = EOS_TOKEN_ID
print(f"Tokenizer Pad Token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")

# --- 3. 加载模型 ---
print(f"从检查点加载模型: {CHECKPOINT_PATH}")
if not os.path.exists(CHECKPOINT_PATH):
    print(f"错误: 检查点路径 {CHECKPOINT_PATH} 不存在！请验证路径。")
    exit()

model_config = AutoConfig.from_pretrained(CHECKPOINT_PATH) # trust_remote_code not used by AutoConfig
model = Qwen2VLForConditionalGeneration.from_pretrained(
    CHECKPOINT_PATH,
    config=model_config,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True,
    # low_cpu_mem_usage=True # Add this if you encounter OOM during model loading
)

# --- 4. 调整模型词嵌入层大小 ---
current_tokenizer_vocab_size = len(tokenizer)
if model.config.vocab_size != current_tokenizer_vocab_size:
    print(f"警告: 模型词汇表大小 ({model.config.vocab_size}) 与Tokenizer词汇表大小 ({current_tokenizer_vocab_size}) 不匹配。")
    print(f"正在调整模型词汇表大小以匹配当前Tokenizer。")
    model.resize_token_embeddings(current_tokenizer_vocab_size)
    model.config.vocab_size = current_tokenizer_vocab_size
else:
    print(f"模型词汇表大小 ({model.config.vocab_size}) 与 Tokenizer ({current_tokenizer_vocab_size}) 一致。")

model.to(DEVICE)
model.eval()
print("模型加载完成并设置为评估模式。")

# --- ANLS 计算 ---
def normalize_text(text: str) -> str:
    """简单的文本规范化。"""
    text = text.lower()
    # text = re.sub(r'[^\w\s]', '', text) # Example: remove punctuation
    text = text.strip()
    return text

def anls_score(prediction: str, ground_truths: List[str]) -> float:
    """计算单个预测与多个真实答案之间的ANLS。"""
    if not ground_truths:
        return 0.0

    normalized_prediction = normalize_text(prediction)
    max_anls = 0.0

    for gt in ground_truths:
        normalized_gt = normalize_text(gt)
        
        if not normalized_gt:
            current_anls = 1.0 if not normalized_prediction else 0.0
        elif not normalized_prediction:
            current_anls = 0.0
        else:
            distance = Levenshtein.distance(normalized_prediction, normalized_gt)
            max_len = max(len(normalized_prediction), len(normalized_gt))
            if max_len == 0:
                nls = 0.0
            else:
                nls = distance / max_len
            current_anls = 1.0 - nls
        
        if current_anls > max_anls:
            max_anls = current_anls
            
    return max_anls

# --- DocVQA 推理函数 ---
def predict_for_docvqa(
    pil_image: PILImage.Image,
    question_text: str,
    model_instance: Qwen2VLForConditionalGeneration,
    processor_instance: AutoProcessor,
    tokenizer_instance: AutoTokenizer
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image.convert("RGB")},
                {"type": "text", "text": f"Question: {question_text}\nAnswer:"}
            ]
        }
    ]
    
    text_prompt = processor_instance.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = processor_instance(
        text=[text_prompt],
        images=[pil_image.convert("RGB")],
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model_instance.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.get('pixel_values'),
            image_grid_thw=inputs.get('image_grid_thw'),
            max_new_tokens=MAX_NEW_TOKENS_DOCVQA,
            eos_token_id=[EOS_TOKEN_ID, tokenizer_instance.eos_token_id],
            pad_token_id=tokenizer_instance.pad_token_id,
            temperature=TEMPERATURE_DOCVQA if TEMPERATURE_DOCVQA > 0 else None,
            top_p=TOP_P_DOCVQA if TEMPERATURE_DOCVQA > 0 else None,
            top_k=TOP_K_DOCVQA if TEMPERATURE_DOCVQA > 0 else None,
            do_sample=True if TEMPERATURE_DOCVQA > 0 else False,
        )

    input_token_len = inputs.input_ids.shape[1]
    generated_ids_trimmed = generated_ids[:, input_token_len:]
    
    output_texts = processor_instance.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_texts[0].strip() if output_texts else ""


# --- DocVQA 评估主函数 ---
def evaluate_docvqa_performance():
    print(f"开始在 DocVQA (数据集: {DOCVQA_DATASET_NAME}, 配置: {DOCVQA_CONFIG_NAME}, Split: {DOCVQA_SPLIT}) 上评估...")
    
    current_split_dataset = load_dataset(
        DOCVQA_DATASET_NAME,
        name=DOCVQA_CONFIG_NAME,
        split=DOCVQA_SPLIT,
        cache_dir=HF_CACHE_DIR
    )
    dataset_length = len(current_split_dataset)
    print(f"数据集加载成功，条目数: {dataset_length}")

    all_anls_scores = []
    predictions_output = []

    for i, example in enumerate(current_split_dataset):
        if (i + 1) % 10 == 0: # Log progress every 10 samples
            print(f"  正在处理第 {i + 1}/{dataset_length} 个问题...")

        question_id = example.get("questionId", f"q_{i}")
        question_text = example["question"]
        ground_truths = example["answers"]
        print(question_id, question_text, ground_truths)

        # Image loading and processing
        # This part will crash if an image is problematic or 'image' field is not as expected
        if isinstance(example['image'], PILImage.Image):
            pil_image = example['image']
        elif isinstance(example['image'], HFImage):
            pil_image = example['image'].convert("RGB")
        elif isinstance(example['image'], dict) and 'bytes' in example['image'] and example['image']['bytes']:
            pil_image = PILImage.open(io.BytesIO(example['image']['bytes'])).convert("RGB")
        # Add elif for local image paths if your dataset structure uses them
        # elif isinstance(example['image'], str) and DOCVQA_IMAGE_DIR:
        #     image_path = os.path.join(DOCVQA_IMAGE_DIR, example['image']) if not os.path.isabs(example['image']) else example['image']
        #     pil_image = PILImage.open(image_path).convert("RGB")
        else:
            # This case might not be reached if a problematic 'image' field already caused an error
            print(f"警告: 问题 {question_id} 图像格式无法直接处理或字段不存在/错误。类型: {type(example['image'])}。跳过。")
            # To strictly remove try-except, this 'continue' implies an unhandled error would have occurred
            # For this script to continue, we'd assume 'example['image']' exists and is one of the above types.
            # If not, an AttributeError or TypeError would likely occur when 'convert' or 'open' is called.
            # If you want to truly skip on unknown format without error, a more complex check or a try-except is needed here.
            # For now, if it's not one of the known types, it will likely error out on the next line trying to use pil_image
            # or if pil_image is not defined. For simplicity, let's assume it's one of the valid types.
            # If you want to strictly skip bad entries without try-except, you'd need to check if 'image' exists and is valid *before* this block.
            # For now, we'll let it proceed, and it might crash if pil_image doesn't get assigned.
            # A more robust "skip" without try-except for unhandled formats:
            pil_image = None
            if isinstance(example.get('image'), PILImage.Image): # check if key exists
                pil_image = example['image'].convert("RGB") # Convert even if it's already PIL, to ensure RGB
            elif isinstance(example.get('image'), HFImage):
                 pil_image = example['image'].convert("RGB")
            # ... (add other checks similarly with .get('image'))
            if pil_image is None:
                 print(f"警告: 问题 {question_id} 图像无法加载或格式未知。跳过。")
                 continue # Skip this sample

        pil_image = pil_image.convert("RGB") # Ensure it's RGB after all checks
        pil_image = pil_image.resize((512, 512))
        model_prediction = predict_for_docvqa(
            pil_image, question_text, model, processor, tokenizer
        )
        
        # If predict_for_docvqa raises an exception, the script will stop here.
        current_anls = anls_score(model_prediction, ground_truths)
        
        all_anls_scores.append(current_anls)
        predictions_output.append({
            "questionId": question_id,
            "question": question_text,
            "ground_truths": ground_truths,
            "prediction": model_prediction,
            "anls": current_anls
        })

        if (i + 1) % 50 == 0 and all_anls_scores: # Log average ANLS every 50 samples
            avg_anls_so_far = sum(all_anls_scores) / len(all_anls_scores)
            print(f"  到目前为止 ({i+1} samples) 的平均 ANLS: {avg_anls_so_far:.4f}")

    if all_anls_scores:
        overall_average_anls = sum(all_anls_scores) / len(all_anls_scores)
        print(f"\n--- DocVQA ({DOCVQA_SPLIT} split) 评估完成 ---")
        print(f"整体平均 ANLS: {overall_average_anls:.4f}")
    else:
        print("没有成功评估任何问题。")
        overall_average_anls = 0.0

    results_filename = f"docvqa_{DOCVQA_SPLIT}_predictions_{os.path.basename(CHECKPOINT_PATH)}.json"
    # Simplified output directory logic
    output_dir_base = os.path.dirname(CHECKPOINT_PATH) if CHECKPOINT_PATH != MODEL_BASE_PATH else "."
    output_dir_for_results = os.path.join(output_dir_base, "docvqa_results_logs") # Ensure this dir makes sense for you
    
    os.makedirs(output_dir_for_results, exist_ok=True)
    output_file_path = os.path.join(output_dir_for_results, results_filename)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint_path": CHECKPOINT_PATH,
            "dataset_name": DOCVQA_DATASET_NAME,
            "dataset_config": DOCVQA_CONFIG_NAME,
            "dataset_split": DOCVQA_SPLIT,
            "average_anls": overall_average_anls,
            "num_samples_evaluated": len(all_anls_scores),
            "individual_predictions": predictions_output
        }, f, ensure_ascii=False, indent=4)
    print(f"DocVQA 评估结果已保存到: {output_file_path}")

    return overall_average_anls

# --- 主执行块 ---
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64" # Optional: manage GPU memory fragmentation

    if not os.path.exists(MODEL_BASE_PATH):
        print(f"错误: 基础模型路径 {MODEL_BASE_PATH} 不存在！")
    elif not os.path.exists(CHECKPOINT_PATH):
         print(f"错误: 模型检查点路径 {CHECKPOINT_PATH} 不存在！")
    else:
        evaluate_docvqa_performance()