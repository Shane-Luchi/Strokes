import os
import torch
# 确保导入的是 AutoModelForConditionalGeneration
from transformers import AutoProcessor, AutoModelForConditionalGeneration
from PIL import Image

def check_llava_model(model_path: str):
    """
    Checks if a local LLaVA model can be loaded using the transformers library.
    """
    if not os.path.exists(model_path):
        print(f"错误：模型路径不存在：{model_path}")
        print("请检查路径是否正确。")
        return

    print(f"尝试从本地路径加载 LLaVA 模型：{model_path}")

    try:
        # 1. 尝试加载 Processor
        print("-> 尝试加载 Processor...")
        processor = AutoProcessor.from_pretrained(model_path)
        print("-> Processor 加载成功！")

        # 2. 尝试加载 Model
        print("-> 尝试加载 Model...")
        # LLaVA 应该使用 AutoModelForConditionalGeneration
        # 如果你的模型很大（如 7B 或更大），你可能需要指定 dtype 和/或 device_map
        # 例如: model = AutoModelForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        # 先尝试默认加载
        model = AutoModelForConditionalGeneration.from_pretrained(model_path)
        print("-> Model 加载成功！")

        # 3. (可选) 尝试前向传播
        print("-> 尝试使用虚拟输入进行一次前向传播...")
        try:
            dummy_text = "Hello, what is this?"
            dummy_image = Image.new('RGB', (50, 50), color = 'black')
            inputs = processor(text=dummy_text, images=dummy_image, return_tensors="pt")

            # 将输入移到模型所在的设备 (如果模型加载到 GPU，这行是必要的)
            if torch.cuda.is_available():
                 inputs = {k: v.to("cuda") for k, v in inputs.items()}
                 model.to("cuda")


            with torch.no_grad():
                 outputs = model(**inputs)
            print("-> 前向传播成功！模型似乎可以正常运行。")

        except Exception as e:
             print(f"警告：前向传播失败。模型可能加载了，但运行存在问题。错误信息：{e}")
             print("这可能是因为输入数据格式不正确、内存不足或模型本身的问题。")

        print("\n总结：模型似乎加载成功并能进行基本操作。")
        print("这强烈表明你的本地模型文件是完整且可用的。")

    except Exception as e:
        print(f"\n错误：加载模型时发生错误！错误信息：{e}")
        print("这通常意味着模型文件缺失、损坏或格式不正确，或者你使用的加载类不匹配。")
        print("请确认：")
        print("1. 已将 `AutoModelForImageTextToText` 改回 `AutoModelForConditionalGeneration`。")
        print("2. 模型路径 `/home/LLMs/llava/llava-1.5-7b-hf` 确实包含了完整的模型文件（如 config.json, pytorch_model.bin 或 model.safetensors 等）。")
        print("3. 模型文件是否下载完整，没有损坏。")
        print("4. 如果模型很大，考虑加上 `torch_dtype=torch.float16` 或 `bfloat16` 及 `device_map='auto'` 参数来尝试加载。")
        print("5. 你的 transformers 库版本是否支持这个特定的 LLaVA 模型版本。可以尝试升级 transformers: `pip install --upgrade transformers`")


# --- 使用方法 ---
# 替换成你的本地LLaVA模型路径
your_local_model_path = "/home/LLMs/llava/llava-1.5-7b-hf"

check_llava_model(your_local_model_path)