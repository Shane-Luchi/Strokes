# src/inference.py
from PIL import Image
import os
import torch
from config import PIC_DIR, MAX_NEW_TOKENS, IMAGE_SIZE

def predict_stroke_order(model, processor, hanzi=None, image_path=None):
    if not hanzi and not image_path:
        raise ValueError("Either 'hanzi' or 'image_path' must be provided.")
    
    input_text = f"汉字 '{hanzi}' 的笔画顺序是？" if hanzi else "请根据图片预测汉字笔画顺序"
    if image_path:
        if not os.path.isabs(image_path):
            image_path = os.path.join(PIC_DIR, os.path.basename(image_path))
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path {image_path} does not exist.")
        img = Image.open(image_path).convert('RGB')
        # Validate image size
        if img.size != IMAGE_SIZE:
            raise ValueError(f"Image {image_path} has size {img.size}, expected {IMAGE_SIZE}")
    else:
        img = None
    
    inputs = processor(
        text=[input_text],
        images=[img] if img else None,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    if 'pixel_values' in inputs:
        inputs['image_grid_thw'] = torch.tensor([1, IMAGE_SIZE[0], IMAGE_SIZE[1]], dtype=torch.long)  # Updated for 200x200
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pixel_values=inputs.get('pixel_values', None),
        image_grid_thw=inputs.get('image_grid_thw', None),
        max_new_tokens=MAX_NEW_TOKENS
    )
    return processor.decode(outputs[0], skip_special_tokens=True)