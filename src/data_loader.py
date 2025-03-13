# src/data_loader.py
from datasets import load_dataset, Features, Sequence, Value
from PIL import Image
from transformers import Qwen2VLProcessor
import os
import torch
from config import PIC_DIR, IMAGE_SIZE

def load_and_preprocess_data(csv_path, processor):
    # Define features to support multi-dimensional arrays
    features = Features({
        'input_ids': Sequence(Value('int64')),
        'attention_mask': Sequence(Value('int64')),
        'pixel_values': Sequence(Value('float32')),  # Support multi-dimensional float arrays
        'image_grid_thw': Sequence(Value('int64')),
        'labels': Sequence(Value('int64'))
    })

    # Load dataset without assuming a header
    dataset = load_dataset('csv', data_files=csv_path, column_names=['hanzi', 'stroke_order', 'id', 'ucs', 'image_path'])
    
    # Verify the number of columns matches the expected structure
    if len(dataset['train'][0]) != 5:
        raise ValueError(f"CSV file {csv_path} does not have the expected 5 columns: {['hanzi', 'stroke_order', 'id', 'ucs', 'image_path']}")

    def preprocess(examples):
        inputs = []
        for hanzi, stroke, img_path in zip(examples['hanzi'], examples['stroke_order'], examples['image_path']):
            # Handle image loading with error handling
            absolute_img_path = os.path.join(PIC_DIR, os.path.basename(img_path))
            try:
                img = Image.open(absolute_img_path).convert('RGB')
                # Validate image size
                if img.size != IMAGE_SIZE:
                    raise ValueError(f"Image {absolute_img_path} has size {img.size}, expected {IMAGE_SIZE}")
            except Exception as e:
                print(f"Error loading image {absolute_img_path}: {e}")
                continue
            
            text = f"汉字 '{hanzi}' 的笔画顺序是？"
            inputs.append({'text': text, 'image': img, 'label': stroke})
        
        if not inputs:
            return None  # Skip empty batches
        
        # Process text and images
        encodings = processor(
            text=[i['text'] for i in inputs],
            images=[i['image'] for i in inputs],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Process labels (stroke orders)
        label_encodings = processor(
            text=[i['label'] for i in inputs],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get the expected shape of pixel_values
        batch_size = len(inputs)
        expected_shape = encodings['pixel_values'].shape[1:]  # e.g., (3, 200, 200) or (num_patches, embed_dim)
        pixel_values = encodings['pixel_values'].view(batch_size, -1).tolist()  # Flatten for storage, will reshape later
        
        return {
            'input_ids': encodings['input_ids'].tolist(),
            'attention_mask': encodings['attention_mask'].tolist(),
            'pixel_values': pixel_values,
            'image_grid_thw': [[1, IMAGE_SIZE[0], IMAGE_SIZE[1]] for _ in range(batch_size)],
            'labels': label_encodings['input_ids'].tolist()
        }
    
    # Map preprocessing with custom features
    tokenized_dataset = dataset['train'].map(
        preprocess,
        batched=True,
        batch_size=5,
        features=features,
        remove_columns=['hanzi', 'stroke_order', 'id', 'ucs', 'image_path'],
        desc="Preprocessing dataset"
    ).filter(lambda x: x is not None)  # Remove failed batches
    
    # Split into train/validation/test
    train_test = tokenized_dataset.train_test_split(test_size=0.2)
    train_val = train_test['train'].train_test_split(test_size=0.125)
    return {
        'train': train_val['train'],
        'validation': train_val['test'],
        'test': train_test['test']
    }