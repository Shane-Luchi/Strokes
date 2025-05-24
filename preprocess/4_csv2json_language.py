import pandas as pd
import json
import os

# 笔画映射字典
stroke_mapping = {
    "1": "横",
    "2": "竖",
    "3": "撇",
    "4": "捺",
    "5": "横折钩"
}

# 读取datasets_with_paths.csv
csv_file = "../data/datasets_with_paths_fonts.csv"
try:
    df = pd.read_csv(csv_file, encoding="utf-8")
except Exception as e:
    print(f"Error reading {csv_file}: {e}")
    exit(1)

# 初始化数据列表
train_data = []
skipped_rows = []
conversation_id = 1  # 用于生成唯一 id

# 为每个汉字生成5个条目
for row in df.itertuples():
    character = row[1]  # 第二列：汉字
    stroke_order = row[2]  # 第三列：笔顺
    image_paths = row[5].split(",")  # 第六列：image_paths
    
    # 确保image_paths包含5个路径
    if len(image_paths) != 5:
        print(f"Warning: {character} has {len(image_paths)} image paths, expected 5")
        skipped_rows.append((character, image_paths))
        continue
    
    # 替换笔顺数字为中文名称
    try:
        # 将笔顺字符串按空格拆分，替换数字，再合并
        stroke_list = str(stroke_order).split()
        stroke_names = [stroke_mapping.get(stroke, stroke) for stroke in stroke_list]
        formatted_stroke_order = " ".join(stroke_names)
    except Exception as e:
        print(f"Warning: Invalid stroke number in {character}: {stroke_order}")
        skipped_rows.append((character, stroke_order))
        continue


    entry = {
        "id": f"identity_{conversation_id}",  # 添加唯一 id
        "conversations": [
            {
                "role": "user",
                "value": f"{character}"
            },
            {
                "role": "assistant",
                "value": formatted_stroke_order
            }
        ]
    }
    train_data.append(entry)
    conversation_id += 1  # 递增 id

# 保存为train_data_augmented.json
output_file = "../data/train_data_language.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

# 打印统计信息
print(f"Generated {len(train_data)} entries in {output_file}")
print(f"Expected 8105 = {8105 } entries")
if skipped_rows:
    print(f"Skipped rows due to invalid data:")
    for char, data in skipped_rows:
        print(f"Character {char}: {data}")