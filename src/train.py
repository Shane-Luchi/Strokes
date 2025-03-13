# src/train.py
from transformers import TrainingArguments, Trainer
import torch
from config import OUTPUT_DIR, BATCH_SIZE, IMAGE_SIZE

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Convert lists back to tensors
        device = model.device
        batch_size = len(inputs['input_ids'])
        
        inputs = {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long).to(device),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long).to(device),
            'pixel_values': torch.tensor(inputs['pixel_values'], dtype=torch.float).view(batch_size, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device),  # Reshape to [batch_size, 3, 200, 200]
            'image_grid_thw': torch.tensor(inputs['image_grid_thw'], dtype=torch.long).to(device),
            'labels': torch.tensor(inputs['labels'], dtype=torch.long).to(device)
        }
        labels = inputs.pop('labels')
        image_grid_thw = inputs.pop('image_grid_thw')
        
        # Ensure image_grid_thw is in the correct shape [batch_size, 3]
        if image_grid_thw.dim() == 2 and image_grid_thw.shape[1] == 3:
            pass
        else:
            image_grid_thw = torch.tensor([[1, IMAGE_SIZE[0], IMAGE_SIZE[1]]] * batch_size, dtype=torch.long).to(device)
        
        # Debug shapes
        print("Input shapes:")
        print(f"  input_ids: {inputs['input_ids'].shape}")
        print(f"  attention_mask: {inputs['attention_mask'].shape}")
        print(f"  pixel_values: {inputs['pixel_values'].shape}")
        print(f"  image_grid_thw: {image_grid_thw.shape}")
        print(f"  labels: {labels.shape}")
        
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
            image_grid_thw=image_grid_thw,
            labels=labels
        )
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def train_model(model, dataset, processor, output_dir=OUTPUT_DIR):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        processing_class=processor
    )
    trainer.train()
    return trainer