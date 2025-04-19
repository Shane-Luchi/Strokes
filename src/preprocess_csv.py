import os
import json
import pandas as pd

# 定义路径
CSV_PATH = '/home/zsy/GithubCode/Strokes/data/datasets_with_paths.csv'
PIC_DIR = '/home/zsy/GithubCode/Strokes/data/pic/'
OUTPUT_JSON_PATH = '/home/zsy/GithubCode/Strokes/data/train_data.json'

# 笔画编号到名称的映射
stroke_map = {
    "1": "横",
    "2": "竖",
    "3": "撇",
    "4": "捺",
    "5": "横折钩"
    # 如果有更多笔画类型，请补充完整
}

def preprocess_csv_to_json(csv_path, output_json_path):
    # 读取 CSV 文件，无列名
    df = pd.read_csv(csv_path, header=None)
    conversations = []
    
    for i, row in df.iterrows():
        char = row[0]  # 汉字
        strokes = row[1].split()  # 笔画编号列表
        stroke_text = " ".join(stroke_map[stroke] for stroke in strokes)  # 转换为笔画名称
        image_path = row[4]  # 图片路径
        # 确保图片路径是相对于 PIC_DIR 的相对路径
        image_rel_path = os.path.join("pic", os.path.basename(image_path))
        
        # 构建对话格式
        conversations.append({
            "id": f"identity_{i+1}",
            "conversations": [
                {
                    "from": "user",
                    "value": f"请你用中文描述图片中汉字的笔画顺序<|vision_start|>{image_rel_path}<|vision_end|>"
                },
                {
                    "from": "assistant",
                    "value": stroke_text
                }
            ]
        })
    
    # 保存为 JSON 文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    print(f"Data saved to {output_json_path}")
    return conversations

if __name__ == "__main__":
    preprocess_csv_to_json(CSV_PATH, OUTPUT_JSON_PATH)