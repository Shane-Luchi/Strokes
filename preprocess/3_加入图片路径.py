import pandas as pd
import os

# 读取 CSV 文件
# 假设你的 CSV 文件名为 'data.csv'，请根据实际情况修改
csv_file = "datasets.csv"
df = pd.read_csv(csv_file, header=None)  # 因为你没提到有表头，我假设没有表头


df.columns = ["汉字", "笔顺", "名称", "UCS"]

image_folder = "pic"
df["image_path"] = df["汉字"].apply(lambda x: os.path.join(image_folder, f"{x}.png"))

# 保存新的 CSV 文件
# 可以选择覆盖原文件或保存为新文件
output_file = "datasets_with_paths.csv"  # 新文件名
# output_file = csv_file  # 如果要覆盖原文件，取消这行注释
df.to_csv(output_file, index=False, header=False)  # 不保存索引和表头

print(f"已处理完成，新文件保存为: {output_file}")
print("前5行预览:")
print(df.head())
