# config.py
CSV_PATH = '/home/zsy/GithubCode/Strokes/data/datasets_with_paths_10.csv'
MODEL_NAME = '/home/LLMs/Qwen/Qwen2-VL-2B-Instruct'
PIC_DIR = '/home/zsy/GithubCode/Strokes/data/pic/'
OUTPUT_DIR = '/home/zsy/GithubCode/Strokes/results'
BATCH_SIZE = 2  # Safer default for multimodal models
MAX_NEW_TOKENS = 20  # Allow for longer stroke sequences
IMAGE_SIZE = (200, 200)  # Updated to match your dataset