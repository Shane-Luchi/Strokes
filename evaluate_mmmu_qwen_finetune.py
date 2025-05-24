'''

CUDA_VISIBLE_DEVICES=3 python evaluate_mmmu_qwen_finetune.py 

CUDA_VISIBLE_DEVICES=3 nohup python evaluate_mmmu_qwen_finetune.py  > evaluate_mmmu_qwen_finetune_521.log 2>&1 &

line337:
        results_filename = f"mmmu_val_eval_subjects_ckpt_1100.json"

'''

import torch
from datasets import load_dataset, Image as HFImage # To load images from dataset
from PIL import Image as PILImage # For type hinting and potentially processing
from modelscope import AutoTokenizer # 与Qwen生态更匹配
from transformers import (
    AutoProcessor, # 或 Qwen2VLProcessor
    Qwen2VLForConditionalGeneration,
    AutoConfig
)
import json
import os
import re
from typing import List, Dict, Any
import io # 用于从bytes加载图像
import ast # 用于安全解析字符串形式的列表

# --- 自定义工具导入 (请确保此文件在您的PYTHONPATH中或同目录下) ---
# 假设 qwen_vl_utils.py 文件与此脚本在同一目录或PYTHONPATH中
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("错误: 无法导入 qwen_vl_utils.py。请确保该文件存在且路径正确。")
    print("qwen_vl_utils.py 需要包含 process_vision_info 函数用于处理图像。")
    exit()


# --- 配置参数 ---
MODEL_BASE_PATH = "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/"  # 您的基础模型路径
CHECKPOINT_PATH = "./output_levenshtein/Qwen2-VL-2B/checkpoint-1000"  # 您要评估的检查点路径
# CHECKPOINT_PATH = "/home/LLMs/Qwen/Qwen2-VL-2B-Instruct/" 
MMMU_DATASET_PATH = "MMMU/MMMU"  # Hugging Face上的MMMU数据集路径
MMMU_SPLIT = "validation"       # 指定评估 'validation' 集

# --- 定义MMMU科目 ---
# 这是从错误消息中获取的完整科目列表 (或您希望评估的科目)
ALL_MMMU_AVAILABLE_SUBJECTS = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
    'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
    'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage',
    'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy',
    'Physics', 'Psychology', 'Public_Health', 'Sociology'

]

# 您可以选择评估所有科目，或者只评估一部分进行快速测试
# 设置为 ALL_MMMU_AVAILABLE_SUBJECTS 可评估所有列出的科目
# 设置为一个子列表，例如 ['Computer_Science', 'Art_Theory', 'Biology'] 来评估特定科目
MMMU_SUBJECTS_TO_EVALUATE = ALL_MMMU_AVAILABLE_SUBJECTS # <--- 修改这里来选择科目
# MMMU_SUBJECTS_TO_EVALUATE = ['Architecture_and_Engineering', 'Math','Computer_Science', 'Economics', 'Electronics' , 'Finance', 'Geography',
#                              'Mechanical_Engineering', 'Physics'] # 例如，仅测试这三个

# --- 自定义缓存目录 for MMMU ---
MMMU_CUSTOM_CACHE_DIR = "data/MMMU_hf_cache"
os.makedirs(MMMU_CUSTOM_CACHE_DIR, exist_ok=True)
print(f"MMMU 数据集将使用缓存目录: {os.path.abspath(MMMU_CUSTOM_CACHE_DIR)}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS_MMMU = 20
TEMPERATURE_MMMU = 0.1
TOP_P_MMMU = 0.9
TOP_K_MMMU = 40

EOS_TOKEN_STRING = "<EOS>"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
# --- 1. 加载 Tokenizer 和 Processor (从基础模型路径) ---
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
)

# --- 2. 处理自定义 EOS Token ---
num_added_tokens = tokenizer.add_special_tokens({'eos_token': EOS_TOKEN_STRING})
if num_added_tokens > 0:
    print(f"已将 '{EOS_TOKEN_STRING}' 添加或指定为 eos_token。新添加到词汇表的 token 数量: {num_added_tokens}")
else:
    print(f"'{EOS_TOKEN_STRING}' 已被配置为 eos_token (可能之前已存在或被更新，未新增token到词汇表)。")
EOS_TOKEN_ID = tokenizer.eos_token_id
current_eos_token_string = tokenizer.eos_token
print(f"Tokenizer 当前的 eos_token 是: '{current_eos_token_string}', 其 ID 是: {EOS_TOKEN_ID}")

# --- 3. 处理 Pad Token ---
if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
    print(f"Tokenizer 的 pad_token 未设置。将 pad_token 设置为与 eos_token相同。")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Tokenizer Pad Token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")

# --- 4. 加载模型 (从微调检查点路径) ---
print(f"从检查点加载模型: {CHECKPOINT_PATH}")
checkpoint_config = AutoConfig.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
print(f"检查点 ({CHECKPOINT_PATH}) 中的 config.json 内的 vocab_size: {checkpoint_config.vocab_size}")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    CHECKPOINT_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# --- 5. 调整模型词嵌入层大小 (关键步骤) ---
if checkpoint_config.vocab_size != len(tokenizer):
    print(f"警告->调整: 模型检查点的 vocab_size ({checkpoint_config.vocab_size}) 与当前 Tokenizer 的词汇表大小 ({len(tokenizer)}) 不匹配。")
    print(f"正在调整模型词汇表大小以匹配当前 Tokenizer。")
    model.resize_token_embeddings(len(tokenizer))
elif model.config.vocab_size != len(tokenizer): # 双重检查，确保加载后的model对象的config也更新
    print(f"警告->调整: 加载后模型的 model.config.vocab_size ({model.config.vocab_size}) 与当前 Tokenizer 的词汇表大小 ({len(tokenizer)}) 不匹配。")
    print(f"正在调整模型词汇表大小以匹配当前 Tokenizer。")
    model.resize_token_embeddings(len(tokenizer))
else:
    print(f"模型词汇表大小 ({model.config.vocab_size}) 与 Tokenizer 词汇表大小 ({len(tokenizer)}) 一致。无需调整。")

model.to(DEVICE)
model.eval()
print("模型加载完成并设置为评估模式。")


# --- MMMU 推理函数 ---
def predict_for_mmmu(
    question_text: str,
    options_list_str_or_list: Any,
    images_pil_list: List[PILImage.Image],
    model_instance: Qwen2VLForConditionalGeneration,
    processor_instance: AutoProcessor,
    tokenizer_instance: AutoTokenizer
):
    parsed_options = []
    if isinstance(options_list_str_or_list, str):
        try:
            options_eval = ast.literal_eval(options_list_str_or_list)
            if isinstance(options_eval, list):
                parsed_options = [str(opt) for opt in options_eval]
            else:
                parsed_options = [opt.strip() for opt in options_list_str_or_list.split('\n') if opt.strip()]
        except (ValueError, SyntaxError):
             parsed_options = [opt.strip() for opt in options_list_str_or_list.split('\n') if opt.strip()]
    elif isinstance(options_list_str_or_list, list):
        parsed_options = [str(opt) for opt in options_list_str_or_list]
    else:
        return "Error: Could not parse options."

    if not parsed_options:
        return "Error: No options found after parsing."

    prompt_str = f"The following is a multiple-choice question. Please read the question, examine the image(s) if any, and choose the correct option from A, B, C, D, etc.\n\nQuestion: {question_text}\n\nOptions:\n"
    for i, opt_text in enumerate(parsed_options):
        option_label = chr(ord('A') + i)
        if not re.match(r"^[A-Z][.)\s:]", opt_text.strip()):
            prompt_str += f"{option_label}) {opt_text}\n"
        else:
            prompt_str += f"{opt_text}\n"
    prompt_str += "\nAnswer:"

    content_list = []
    for img_pil in images_pil_list:
         content_list.append({"type": "image", "image": img_pil})
    content_list.append({"type": "text", "text": prompt_str})
    messages = [{"role": "user", "content": content_list}]
    
    text_for_processor = processor_instance.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs_for_processor, _ = process_vision_info(messages)

    inputs = processor_instance(
        text=[text_for_processor], images=image_inputs_for_processor,
        videos=None, padding=True, return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model_instance.generate(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
            pixel_values=inputs.get('pixel_values'), image_grid_thw=inputs.get('image_grid_thw'),
            max_new_tokens=MAX_NEW_TOKENS_MMMU,
            eos_token_id=[EOS_TOKEN_ID, tokenizer_instance.eos_token_id],
            pad_token_id=tokenizer_instance.pad_token_id,
            temperature=TEMPERATURE_MMMU, top_p=TOP_P_MMMU, top_k=TOP_K_MMMU,
            do_sample=True if TEMPERATURE_MMMU > 0 else False,
        )

    input_token_len = inputs.input_ids.shape[1]
    generated_ids_trimmed = generated_ids[:, input_token_len:]
    output_text_list = processor_instance.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return output_text_list[0].strip() if output_text_list else ""


def parse_mmmu_model_answer(model_output_text: str, options_count: int):
    model_output_text = model_output_text.strip().upper()
    possible_options_letters = [chr(ord('A') + i) for i in range(options_count)]
    option_chars_regex_part = "".join(possible_options_letters)
    match = re.match(rf"^\s*([{option_chars_regex_part}])(?:[.)\s]|$)", model_output_text)
    if match: return match.group(1)
    if model_output_text in possible_options_letters: return model_output_text
    first_word_match = re.match(r"^\s*(\w+)", model_output_text)
    if first_word_match:
        first_word = first_word_match.group(1)
        if first_word in possible_options_letters: return first_word
    for opt_char in possible_options_letters:
        if re.search(rf"(?:OPTION|ANSWER)[^A-Z0-9]*{opt_char}(?:[.)\s]|$)", model_output_text): return opt_char
        if model_output_text.startswith(f"{opt_char}.") or model_output_text.startswith(f"{opt_char})"): return opt_char
    return None

# --- MMMU 评估主函数 ---
def evaluate_mmmu_performance():
    print(f"开始在 MMMU ({MMMU_SPLIT} split) 上评估以下科目: {MMMU_SUBJECTS_TO_EVALUATE}")
    all_overall_subject_accuracies = {}
    grand_total_correct_all_subjects = 0
    grand_total_questions_all_subjects = 0

    for subject_name in MMMU_SUBJECTS_TO_EVALUATE:
        print(f"\n--- 开始处理科目: {subject_name} ---")
        try:
            print(f"正在加载 MMMU 数据集: {MMMU_DATASET_PATH}, 科目配置: {subject_name}, split: {MMMU_SPLIT}")
            print(f"数据将缓存到: {os.path.abspath(MMMU_CUSTOM_CACHE_DIR)}")
            dataset_iterable = load_dataset(
                MMMU_DATASET_PATH, name=subject_name, split=MMMU_SPLIT,
                trust_remote_code=True, cache_dir=MMMU_CUSTOM_CACHE_DIR
            )
            dataset_length = len(dataset_iterable) if hasattr(dataset_iterable, '__len__') else '未知'
            print(f"科目 '{subject_name}' 加载成功，条目数: {dataset_length}")
        except Exception as e:
            print(f"加载 MMMU 科目 '{subject_name}' 失败: {e}\n请确保科目名有效。跳过此科目。")
            continue

        current_subject_correct = 0
        current_subject_total_questions = 0
        total_examples_in_subject = len(dataset_iterable) if hasattr(dataset_iterable, '__len__') else None

        for i, example in enumerate(dataset_iterable):
            if (i + 1) % 10 == 0:
                progress_str = f"第 {i + 1}"
                if total_examples_in_subject: progress_str += f"/{total_examples_in_subject}"
                print(f"  科目 '{subject_name}': 正在处理 {progress_str} 个问题...")

            question_id = example.get("id", f"unknown_id_{subject_name}_{i}")
            question_text = example["question"]
            options_raw = example["options"]
            correct_answer_letter = str(example["answer"]).strip().upper()
            
            images_pil = []
            has_image_field = False
            for k in range(1, 8):
                img_field = f"image_{k}"
                if img_field in example and example[img_field]:
                    has_image_field = True
                    img_data = example[img_field]
                    try:
                        if isinstance(img_data, PILImage.Image): images_pil.append(img_data.convert("RGB"))
                        elif isinstance(img_data, dict) and 'bytes' in img_data and img_data['bytes']:
                            images_pil.append(PILImage.open(io.BytesIO(img_data['bytes'])).convert("RGB"))
                        elif isinstance(img_data, str) and os.path.exists(img_data):
                             images_pil.append(PILImage.open(img_data).convert("RGB"))
                    except Exception as e_img: print(f"警告: 问题 {question_id} 图像 {img_field} 加载失败: {e_img}")
            
            if has_image_field and not images_pil:
                print(f"警告: 问题 {question_id} (科目: {subject_name}) 声明了图像但未能加载。跳过。")
                continue
            
            temp_parsed_options = [] # 用于获取选项数量
            if isinstance(options_raw, str):
                try:
                    options_eval = ast.literal_eval(options_raw)
                    if isinstance(options_eval, list): temp_parsed_options = [str(opt) for opt in options_eval]
                    else: temp_parsed_options = [opt.strip() for opt in options_raw.split('\n') if opt.strip()]
                except (ValueError, SyntaxError): temp_parsed_options = [opt.strip() for opt in options_raw.split('\n') if opt.strip()]
            elif isinstance(options_raw, list): temp_parsed_options = [str(opt) for opt in options_raw]
            
            if not temp_parsed_options:
                print(f"错误: 问题 {question_id} (科目: {subject_name}) 选项无法解析。跳过。")
                continue
            options_count = len(temp_parsed_options)

            model_output = predict_for_mmmu(
                question_text, options_raw, images_pil, model, processor, tokenizer
            )
            if model_output.startswith("Error:"):
                print(f"错误: 问题 {question_id} (科目: {subject_name}) 推理失败: {model_output}")
                continue
            parsed_model_answer = parse_mmmu_model_answer(model_output, options_count)

            current_subject_total_questions += 1
            if parsed_model_answer == correct_answer_letter:
                current_subject_correct += 1
        
        if current_subject_total_questions > 0:
            subject_accuracy = (current_subject_correct / current_subject_total_questions) * 100
            all_overall_subject_accuracies[subject_name] = subject_accuracy
            print(f"--- 科目 '{subject_name}' 评估完成 ---\n准确率: {subject_accuracy:.2f}% ({current_subject_correct}/{current_subject_total_questions})")
        else:
            all_overall_subject_accuracies[subject_name] = "N/A (no valid questions)"
            print(f"--- 科目 '{subject_name}' 未评估任何有效问题 ---")

        grand_total_correct_all_subjects += current_subject_correct
        grand_total_questions_all_subjects += current_subject_total_questions

    print("\n======= MMMU 总体评估结果 (所有指定科目) =======")
    for sub_name, acc_val in sorted(all_overall_subject_accuracies.items()):
        if isinstance(acc_val, float): print(f"科目: {sub_name:<40} - 准确率: {acc_val:>6.2f}%")
        else: print(f"科目: {sub_name:<40} - 准确率: {acc_val}")

    if grand_total_questions_all_subjects > 0:
        overall_avg_accuracy = (grand_total_correct_all_subjects / grand_total_questions_all_subjects) * 100
        print(f"\n平均准确率 (所有已评估问题): {overall_avg_accuracy:.2f}%")
        print(f"总计正确数 (所有已评估问题): {grand_total_correct_all_subjects}")
        print(f"总计问题数 (所有已评估问题): {grand_total_questions_all_subjects}")
    else:
        print("没有成功评估任何科目或问题。")
    return all_overall_subject_accuracies, grand_total_correct_all_subjects, grand_total_questions_all_subjects

# --- 主执行块 ---
def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"错误: 检查点路径 {CHECKPOINT_PATH} 不存在！请验证路径。")
        return
    if 'process_vision_info' not in globals() or not callable(globals()['process_vision_info']):
        print("错误: 函数 process_vision_info 未定义或无法调用。请确保 qwen_vl_utils.py 正确导入。")
        return
    print(f"使用设备: {DEVICE}")
    
    mmmu_results, total_correct, total_questions = evaluate_mmmu_performance()

    if mmmu_results: 
        results_filename = f"mmmu_val_all_1000.json"
        # results_filename = f"Qwen2VL-2B-all-subject.json"
        output_dir_for_results = os.path.dirname(CHECKPOINT_PATH)
        # if not output_dir_for_results or not os.path.exists(output_dir_for_results) or not os.path.isdir(output_dir_for_results) :
        output_dir_for_results = "./logs/MMMU"
        output_file_path = os.path.join(output_dir_for_results, results_filename)
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "checkpoint_path": CHECKPOINT_PATH,
                    "mmmu_dataset_path": MMMU_DATASET_PATH,
                    "mmmu_subjects_evaluated_config": MMMU_SUBJECTS_TO_EVALUATE,
                    "mmmu_split": MMMU_SPLIT,
                    "mmmu_custom_cache_dir_used": os.path.abspath(MMMU_CUSTOM_CACHE_DIR),
                    "overall_accuracy_avg_across_evaluated": (total_correct / total_questions * 100) if total_questions > 0 else 0,
                    "total_correct_evaluated": total_correct,
                    "total_questions_evaluated": total_questions,
                    "subject_specific_accuracies": mmmu_results
                }, f, ensure_ascii=False, indent=4)
            print(f"\nMMMU 评估结果已保存到: {output_file_path}")
        except Exception as e:
            print(f"保存结果到JSON文件时出错: {e}")

if __name__ == "__main__":
    main()