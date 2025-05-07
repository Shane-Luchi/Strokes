# Strokes
微调一个汉字笔顺的模型

* 首先从国家网站上了解到了再2020年出版的国家规定，数据集从这个规定上提取。
* 编码问题

*上传到服务器上，开始选择模型进行微调*
Strokes环境用来做主任务


环境有些冲突，所以创建另一个基本环境llava-env，用来做llava的Prompt.py测评，避免环境混乱。
```bash
git clone https://github.com/haotian-liu/LLaVA.git

cd LLaVA

pip install -e .

```
详细可以参考这个[链接](https://github.com/haotian-liu/LLaVA.git)



def predict_llava(image_path):
    print(f"开始处理图像: {image_path}")
    model_path = "/home/LLMs/llava/llava-1.5-7b-hf"
    model_name = get_model_name_from_path(model_path)
    print("开始加载模型...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,  # config_path
        model_name,
        device_map=None  # 显式设置 device_map 为 None
    )
    print("模型加载完成。")
    model.to("cuda")

    print(f"打开图像: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print("图像打开完成。")
    image_tensor = image_processor(image, return_tensors="pt")["pixel_values"].to("cuda")
    print("图像预处理完成。")

    prompt = "请你用中文描述图片中汉字的笔画顺序"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print("开始模型推理...")
    outputs = model.generate(
        input_ids=inputs.input_ids,
        pixel_values=image_tensor,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.9
    )
    print("模型推理完成。")
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"预测结果: {output_text}")
    return output_text