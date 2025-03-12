import tabula
import json
import os

# 设置PDF文件路径
pdf_path = "/Users/zhengshuyan/Downloads/笔顺/2/009-012.pdf"  # 请替换为您的PDF文件路径

# 提取PDF中所有页面的表格
tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)

# 初始化一个空列表，用于存储所有汉字的数据
characters_data = []

# 处理每个表格
for table in tables:
    # 假设表格列名为：汉字、笔顺、《字源》声旁、UCS
    # 如果列名不同，请根据实际提取结果调整
    print("列名:", table.columns.tolist())  # 打印所有列名
    # 打印这几列的前5行数据
    print(table[['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', '《字表》', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', '《字表》.1', 'Unnamed: 8']].head())
    for index, row in table.iterrows():
        # 提取每个字段
        character = row['汉字']
        stroke_order = row['笔顺']
        ucs_code = row['UCS']
        
        # 处理笔顺：将空格替换为逗号，生成逗号分隔的字符串
        # 例如："1 2" -> "1,2"
        stroke_order = stroke_order.replace(" ", ",").replace(",,", ",")
        
        # 生成图片路径
        image_path = f"images/{character}.png"
        
        # 创建汉字的字典
        character_dict = {
            "character": character,
            "stroke_order": stroke_order,
            "image_path": image_path,
            "ucs_code": ucs_code
        }
        
        # 添加到列表中
        characters_data.append(character_dict)

# 设置JSON文件保存路径
json_path = "characters.json"

# 确保images文件夹存在（用于存储图片）
os.makedirs("images", exist_ok=True)

# 将数据写入JSON文件
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(characters_data, json_file, ensure_ascii=False, indent=4)

print(f"JSON文件已生成，保存路径：{json_path}")