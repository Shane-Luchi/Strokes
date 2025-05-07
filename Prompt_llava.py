import json
import os
from PIL import Image
import torch

# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from Levenshtein import distance as levenshtein_distance

# --- 配置区域 ---
MODEL_PATH = "/home/LLMs/llava/llava-1.5-7b-hf"  # 从 Hugging Face Hub 加载
DATA_JSON_PATH = "data/data_vl_test.json"
IMAGE_BASE_PATH = "data/"
# PROMPT_TEXT = "请你用中文描述图片中汉字的笔画顺序，直接给出笔画顺序，从下面五种笔画的范围内选取：横、竖、撇、捺、横折钩。每个笔画之间用空格空开，不要采用顿号，逗号等其他符号。"

QUESTION_TEXT = "请你用中文描述图片中汉字的笔画顺序，直接给出笔画顺序，从下面五种笔画的范围内选取：横、竖、撇、捺、横折钩。每个笔画之间用空格空开，不要采用顿号，逗号等其他符号。"

# A common format for LLaVA 1.5
PROMPT_TEXT = f"USER: <image>\n{QUESTION_TEXT}\nASSISTANT:"



MAX_NEW_TOKENS = 64
# --- 配置区域结束 ---

def load_llava_model(model_path, device):
    """
    从 Hugging Face Hub 加载 LLaVA 模型和处理器。
    """
    print(f"开始从 Hugging Face Hub 加载 LLaVA 模型: {model_path}")
    # try:
    tokenizer = AutoProcessor.from_pretrained(model_path)

    model = AutoModelForImageTextToText.from_pretrained(model_path)
    model.to(device)

    # 如果在 CUDA 上运行，通常建议将模型转换为半精度 (float16)
    # 以匹配输入数据的精度，并提高效率/减少显存占用
    if device.type == 'cuda':
        model.half()  # 将模型的参数和缓冲区转换为 float16
    # <<< --- 添加结束 --- >>>

    print(f"LLaVA 模型 ({model_path}) 加载完成。")
    
    return tokenizer, model
    # except Exception as e:
    #     print(f"加载 LLaVA 模型失败: {e}")
    #     raise

def predict_with_llava(image_path, prompt, tokenizer, model, device):
    """
    使用 LLaVA 模型对单个图像进行预测。
    """
    print(f"\n正在处理图片: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = tokenizer(text=prompt, images=image, return_tensors="pt").to(device,torch.float16)
        print("开始模型推理...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # 使用贪心解码
                use_cache=True,
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # --- 从这里开始添加/修改 ---
        assistant_marker = "ASSISTANT:"
        if assistant_marker in output_text:
            # 找到 "ASSISTANT:" 之后的部分
            generated_strokes = output_text.split(assistant_marker, 1)[-1].strip()
            # 进一步清理可能由模型生成的不期望的前缀或换行（如果ASSISTANT:后面有空格或换行）
            # 例如，如果ASSISTANT: 后面直接就是答案，strip()就够了
            # 如果模型可能生成 ASSISTANT: \n 答案，strip()也能处理
        else:
            # 如果没有找到 "ASSISTANT:" 标记，则可能输出格式有变，或者就是原始的输出
            # 这里可以根据需要决定如何处理，例如直接使用 output_text 或者打印一个警告
            generated_strokes = "未找到ASSISTANT标记，无法提取笔画序列。"
            print(f"  警告: 在输出中未找到 '{assistant_marker}' 标记。")

        print(f"  提取的笔画序列: {generated_strokes}")
        # --- 修改结束 ---

        # 如果希望此函数返回清理后的笔画序列，而不是完整的 output_text
        # return generated_strokes
        return generated_strokes # 当前函数仍然返回完整的 output_text，您可以按需修改






        return output_text

    except FileNotFoundError:
        print(f"  错误: 图片文件未找到 {image_path}")
        return None
    except Exception as e:
        print(f"  处理图片 {image_path} 时发生错误: {e}")
        return None



def main():
    # 设置设备
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    rouge = Rouge()
    results = []
    # 加载模型
    # try:
    llava_tokenizer, llava_model = load_llava_model(MODEL_PATH, device)
    # except Exception:
    #     print("无法加载 LLaVA 模型，程序退出。")
    #     return

    # 加载测试集
    try:
        with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
            test_dataset = json.load(f)
        print(f"测试集大小: {len(test_dataset)}")
    except FileNotFoundError:
        print(f"错误: 测试数据文件 {DATA_JSON_PATH} 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误: 解析测试数据文件 {DATA_JSON_PATH} 失败。")
        return


    total_samples = len(test_dataset)
    correct_samples = 0
    bleu_score_avg = 0
    rouge_score_avg = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
    levenshtein_avg = 0
    processed_samples_count = 0

    # 遍历测试集进行推理
    for i, example in enumerate(test_dataset):
        # if i >= 10:  # 限制处理数量，便于测试
        #     print("\n已达到最大处理样本数（用于测试），提前结束。")
        #     break

        # 从数据集中提取图像路径
        try:
            input_content = example["conversations"][0]["value"]
            relative_image_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
            image_full_path = os.path.join(IMAGE_BASE_PATH, relative_image_path)
        except (IndexError, KeyError) as e:
            print(f"警告: 无法从样本 {i} 中解析图像路径: {input_content[:100]}... 错误: {e}")
            continue

        # 使用 LLaVA 进行预测
        pred_output = predict_with_llava(
            image_path=image_full_path,
            prompt=PROMPT_TEXT,
            tokenizer=llava_tokenizer,
            model=llava_model,
            device=device,
        )
        if pred_output  is None: # 假设predict_with_llava在错误时返回None
            continue
        processed_samples_count += 1 # 在成功获取预测后计数
        

        true_output = example["conversations"][1]["value"].strip()
        input_content = example["conversations"][0]["value"]
        file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
        file_path = f"data/{file_path}"
        chinese_character = os.path.basename(file_path).split(".")[0]
        print(f"处理字符: {chinese_character}")


        # 计算准确率
        if pred_output == true_output:
            correct_samples += 1
        pred_output = pred_output.replace("、", " ")
        print(f"预测: {pred_output}")
        print(f"真实: {true_output}")
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
        levenshtein_dist = levenshtein_distance(pred_output.split(), true_output.split())
        levenshtein_avg += levenshtein_dist
        print(f"Levenshtein 距离: {levenshtein_dist}")
        # 平均指标
    if processed_samples_count > 0:
        bleu_score_avg /= processed_samples_count
        for key in rouge_score_avg:
            rouge_score_avg[key] /= processed_samples_count
        levenshtein_avg /= processed_samples_count
        accuracy = correct_samples / processed_samples_count
    else:
        # 处理没有样本被成功处理的情况
        bleu_score_avg = 0.0
        # ... 其他指标也设为0或NaN
        accuracy = 0.0


    # bleu_score_avg /= total_samples
    # for key in rouge_score_avg:
    #     rouge_score_avg[key] /= total_samples
    # levenshtein_avg /= total_samples
    # accuracy = correct_samples / total_samples if total_samples > 0 else 0
    print({
    "准确率": accuracy,
    "BLEU": bleu_score_avg,
    "ROUGE-1": rouge_score_avg["rouge-1"],
    "ROUGE-2": rouge_score_avg["rouge-2"],
    "ROUGE-L": rouge_score_avg["rouge-l"],
    "Levenshtein": levenshtein_avg,
    "正确样本数": correct_samples,
    "总样本数": total_samples})


if __name__ == "__main__":
    main()
