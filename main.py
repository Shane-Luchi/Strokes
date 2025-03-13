# main.py
import os
import torch
from src.data_loader import load_and_preprocess_data
from src.model import load_model
from src.train import train_model
from src.inference import predict_stroke_order
from src.evaluate import evaluate_model
from transformers import Qwen2VLProcessor
from config import CSV_PATH, MODEL_NAME, PIC_DIR, IMAGE_SIZE, OUTPUT_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # Print system info
    print("可用 GPU 数量:", torch.cuda.device_count())
    print("当前 GPU:", torch.cuda.current_device())
    print("GPU 名称:", torch.cuda.get_device_name(0))
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 版本:", torch.version.cuda)
    print("cuDNN 版本:", torch.backends.cudnn.version())
    print("CUDA 可用:", torch.cuda.is_available())
    
    # Configure processor for 200x200 images
    min_pixels = 256 * IMAGE_SIZE[0] * IMAGE_SIZE[1]  # 256 * 200 * 200 = 10,240,000
    max_pixels = 1280 * IMAGE_SIZE[0] * IMAGE_SIZE[1]  # 1280 * 200 * 200 = 51,200,000
    processor = Qwen2VLProcessor.from_pretrained(MODEL_NAME, min_pixels=min_pixels, max_pixels=max_pixels)
    
    # Load and preprocess data
    dataset = load_and_preprocess_data(CSV_PATH, processor)
    
    # Load model
    model = load_model(MODEL_NAME, use_lora=True, lora_r=16, lora_alpha=32)
    
    # Train model
    trainer = train_model(model, dataset, processor, output_dir=OUTPUT_DIR)
    
    # Evaluate model
    metrics = evaluate_model(model, processor, dataset['test'])
    print("评估结果:", metrics)
    
    # Example predictions
    print("示例预测:")
    print(predict_stroke_order(model, processor, hanzi="一"))
    print(predict_stroke_order(model, processor, image_path="pic/二.png"))
    
    # Save model and processor
    model.save_pretrained(f'{OUTPUT_DIR}/final_model')
    processor.save_pretrained(f'{OUTPUT_DIR}/final_model')

if __name__ == "__main__":
    main()