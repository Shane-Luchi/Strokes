import pandas as pd
import os

# 字体子文件夹名称
font_names = [
    "noto_serif",
    "source_han_sans",
    "AlibabaPuHuiTi",
    "kaiti",
    "hanyi_senty_wen"
]

# 读取CSV文件（无列名）
csv_file = "../datasets.csv"
df = pd.read_csv(csv_file, header=None)

# 假设列顺序为：汉字, 笔顺, 名称, UCS
# 给列命名以便操作
df.columns = ["汉字", "笔顺", "名称", "UCS"]

# 生成图片路径（逗号分隔的字符串）
def generate_image_paths(character):
    paths = [os.path.join("pic", font_name, f"{character}.png") for font_name in font_names]
    return ",".join(paths)

# 添加image_paths列
df["image_paths"] = df["汉字"].apply(generate_image_paths)

# 保存新的CSV文件
output_file = "../data/datasets_with_paths_fonts.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"已处理完成，新文件保存为: {output_file}")
print("前5行预览:")
print(df.head())