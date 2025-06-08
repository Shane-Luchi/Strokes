# 本文件通过处理带5种字体路径的csv文件，转换成json格式，并拆分训练集和验证集
import pandas as pd
import json
import os
import random

# 笔画映射字典
stroke_mapping = {"1": "横", "2": "竖", "3": "撇", "4": "捺", "5": "横折钩"}

# 读取datasets_with_paths.csv
csv_file = "../data/datasets_with_paths_fonts.csv"
try:
    df = pd.read_csv(csv_file, encoding="utf-8")
except Exception as e:
    print(f"Error reading {csv_file}: {e}")
    exit(1)

# 初始化数据列表
train_data = []
valid_data = []
skipped_rows = []
conversation_id = 1  # 用于生成唯一 id

# 为每个汉字生成5个条目
for row in df.itertuples():
    t = []
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

    # 为每种字体生成一个条目
    for image_path in image_paths:
        # 验证图片存在（可选，取消注释以启用）
        # if not os.path.exists(image_path):
        #     print(f"Warning: Image {image_path} does not exist")
        #     continue
        entry = {
            "messages": [
                {
                    "content": f"""
                <image>请严格按照以下五种基本笔画类型（横、竖、撇、捺、横折钩）分解图片中汉字的笔顺，并按书写顺序逐笔列出。  
                要求：
                1. 只使用这五种笔画名称，不细分（如“点”归为“捺”，“竖钩”归为“竖”等）。
                2. 按正确笔顺逐笔输出，输出结果放在一行中，每个笔画之间用一个空格隔开，格式示例：横 竖 撇 捺 横折钩  

                例如「木」字的笔顺：横 竖 撇 捺  

                现在请分析图片：
                """,
                    "role": "user",
                },
                {"content": formatted_stroke_order, "role": "assistant"},
            ],
            "images": [f"{image_path}"],
        }
        t.append(entry)
        conversation_id += 1  # 递增 id
    tmp = random.random()
    if tmp <= 0.9:
        train_data.extend(t)
    else:
        valid_data.extend(t)


# 保存为train_data_augmented.json
output_file_train = "../data/train_data_llama.json"
os.makedirs(os.path.dirname(output_file_train), exist_ok=True)
with open(output_file_train, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

output_file_valid = "../data/valid_data_llama.json"
os.makedirs(os.path.dirname(output_file_valid), exist_ok=True)
with open(output_file_valid, "w", encoding="utf-8") as f:
    json.dump(valid_data, f, ensure_ascii=False, indent=2)

# 打印统计信息
print(f"Generated {len(train_data)} entries in {output_file_train}")
print(f"Generated {len(valid_data)} entries in {output_file_valid}")
print(f"Expected 8105 * 5 = {8105 * 5} entries")
if skipped_rows:
    print(f"Skipped rows due to invalid data:")
    for char, data in skipped_rows:
        print(f"Character {char}: {data}")
