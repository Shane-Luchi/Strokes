# src/evaluate.py
from Levenshtein import distance

def evaluate_model(model, processor, test_dataset):
    predictions = []
    references = []
    for example in test_dataset:
        pred = predict_stroke_order(model, processor, hanzi=processor.decode(example['input_ids'], skip_special_tokens=True).split("'")[1])
        predictions.append(pred)
        references.append(processor.decode(example['labels'], skip_special_tokens=True))
    
    exact_match = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    avg_edit_distance = sum(distance(p, r) for p, r in zip(predictions, references)) / len(predictions)
    return {"exact_match": exact_match, "avg_edit_distance": avg_edit_distance}