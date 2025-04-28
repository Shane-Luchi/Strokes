import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os

# 定义字体路径和对应的子文件夹名称
font_paths = [
    "../fonts/NotoSerifCJKsc-Regular.otf",
    "../fonts/SourceHanSansCN-VF.otf",
    "../fonts/chinese.simfang.ttf",
    "../fonts/simkai.ttf",
    "../fonts/SentyWEN2017.ttf"
]
font_names = [
    "noto_serif",
    "source_han_sans",
    "fangsong",
    "kaiti",
    "hanyi_senty_wen"
]

# 创建图片保存的子文件夹
base_dir = os.path.dirname(__file__)  # 获取脚本所在目录
# pic_dir = os.path.join(base_dir, "pic")
pic_dir = "../data/pic"
for font_name in font_names:
    os.makedirs(os.path.join(pic_dir, font_name), exist_ok=True)

# 读取数据集（无列名，假设列顺序为：汉字, 笔顺, 名称, UCS）
df = pd.read_csv("../datasets.csv", header=None, encoding="utf-8")

# 设置图片和字体大小
img_size = (280, 280)
font_size = 150

# 为每个汉字生成多种字体的图片
for row in df.itertuples():
    # 直接使用第一列（索引0）获取汉字
    character = row[1]
    print(character)
    if not character:  # 跳过空字符
        print("Skipping empty character")
        continue
    
    for i, font_path in enumerate(font_paths):
        font_name = font_names[i]
        font_path_full = os.path.join(base_dir, font_path)
        try:
            # 验证字体文件存在
            if not os.path.exists(font_path_full):
                raise FileNotFoundError(f"Font file {font_path_full} not found")
            
            # 加载字体
            font = ImageFont.truetype(font_path_full, font_size)
            
            # 获取文字的边界框并计算宽度和高度
            bbox = font.getbbox(character)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 计算文字居中位置
            x = (img_size[0] - text_width) / 2
            y = (img_size[1] - text_height) / 2 - bbox[1]
            
            # 创建280x280白色背景图片
            img = Image.new("RGB", img_size, color="white")
            d = ImageDraw.Draw(img)
            
            # 在图片上绘制汉字
            d.text((x, y), character, font=font, fill="black")
            
            # 保存图片到对应字体的子文件夹
            save_path = os.path.join(pic_dir, font_name, f"{character}.png")
            img.save(save_path)
            print(f"Generated image for {character} with font {font_name}")
            
        except Exception as e:
            print(f"Error generating image for {character} with font {font_name}: {e}")
    # exit()