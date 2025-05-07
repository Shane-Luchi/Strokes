import json
import os
from PIL import Image
import torch

# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText
# --- 配置区域 ---
MODEL_PATH = "llava-hf/llava-1.5-7b-hf"  # 从 Hugging Face Hub 加载
DATA_JSON_PATH = "data/data_vl_test.json"
IMAGE_BASE_PATH = "data/"
PROMPT_TEXT = "请你用中文描述图片中汉字的笔画顺序，直接给出笔画顺序，从下面五种笔画的范围内选取：横、竖、撇、捺、横折钩。每个笔画之间用空格空开，不要采用顿号，逗号等其他符号。"
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
        inputs = tokenizer(prompt, images=image, return_tensors="pt").to(device,
                                                                         torch.float16)  # 图像和文本一起处理, 并移动到设备, 使用半精度

        print("开始模型推理...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # 使用贪心解码
                use_cache=True,
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"  原始输入 Prompt: {prompt}")
        print(f"  LLaVA 输出: {output_text}")
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

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

    # 遍历测试集进行推理
    for i, example in enumerate(test_dataset):
        if i >= 5:  # 限制处理数量，便于测试
            print("\n已达到最大处理样本数（用于测试），提前结束。")
            break

        # 从数据集中提取图像路径
        try:
            input_content = example["conversations"][0]["value"]
            relative_image_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
            image_full_path = os.path.join(IMAGE_BASE_PATH, relative_image_path)
        except (IndexError, KeyError) as e:
            print(f"警告: 无法从样本 {i} 中解析图像路径: {input_content[:100]}... 错误: {e}")
            continue

        # 使用 LLaVA 进行预测
        predict_with_llava(
            image_path=image_full_path,
            prompt=PROMPT_TEXT,
            tokenizer=llava_tokenizer,
            model=llava_model,
            device=device,
        )



if __name__ == "__main__":
    main()
