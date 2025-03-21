import os
import json
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch

# Set GPU devices (using 2 A800 GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Define paths and parameters
JSON_PATH = '/home/zsy/GithubCode/Strokes/data/train_data_10.json'
MODEL_NAME = '/home/LLMs/Qwen/Qwen2-VL-2B-Instruct'
PIC_DIR = '/home/zsy/GithubCode/Strokes/data/pic/'
OUTPUT_DIR = '/home/zsy/GithubCode/Strokes/results'

# 1. Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Use FP16 to reduce memory usage
    device_map="auto"
)
processor = Qwen2VLProcessor.from_pretrained(MODEL_NAME)

# 2. Configure LoRA
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target modules (attention layers)
    lora_dropout=0.05,  # Dropout rate
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print trainable parameter count

# 3. Load preprocessed JSON data
def load_json_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

data = load_json_data(JSON_PATH)

# 4. Preprocessing function
def preprocess_function(examples):
    # Load images
    images = [Image.open(os.path.join(PIC_DIR, item["image"].split('/')[-1])).convert("RGB") for item in examples["data"]]
    texts = [item["text"] for item in examples["data"]]
    
    # Construct prompt text
    prompt = "请拆解这个汉字的笔画顺序："
    batch_size = len(images)
    
    # Process prompt text
    text_inputs = processor.tokenizer(
        [prompt] * batch_size,  # Same prompt for each image
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Process target text (stroke order)
    target_inputs = processor.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Process images and calculate grid_thw
    image_size = processor.image_processor.image_size  # e.g., 448
    patch_size = processor.image_processor.patch_size  # e.g., 32
    grid_thw = (image_size // patch_size, image_size // patch_size)  # e.g., (14, 14)
    
    # Process images into pixel values
    pixel_values_list = [processor.image_processor(image, return_tensors="pt")["pixel_values"] for image in images]
    pixel_values = torch.stack(pixel_values_list, dim=0)  # Shape: (batch_size, num_channels, height, width)
    
    # Create grid_thw tensor for the batch
    grid_thw_tensor = torch.tensor([grid_thw] * batch_size, dtype=torch.long)  # Shape: (batch_size, 2)
    
    # Construct model inputs
    inputs = {
        "pixel_values": pixel_values,
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "labels": target_inputs["input_ids"],
        "grid_thw": grid_thw_tensor  # Add grid_thw here
    }
    
    # Verify batch size consistency
    for key, value in inputs.items():
        assert value.shape[0] == batch_size, f"Expected {key} to have batch size {batch_size}, but got shape {value.shape}"
    
    return inputs

# Create and preprocess dataset
dataset = Dataset.from_dict({"data": data})
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["data"])

# 5. Set training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,  # Mixed precision training
    dataloader_num_workers=8,
    gradient_accumulation_steps=1,
    report_to="none"
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 7. Start training
trainer.train()

# 8. Save the model
model.save_pretrained(os.path.join(OUTPUT_DIR, "finetuned_model"))
processor.save_pretrained(os.path.join(OUTPUT_DIR, "finetuned_model"))
print(f"Model saved to {os.path.join(OUTPUT_DIR, 'finetuned_model')}")