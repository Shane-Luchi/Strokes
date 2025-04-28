a = "pic/noto_serif/一.png,pic/source_han_sans/一.png,pic/fangsong/一.png,pic/kaiti/一.png,pic/hanyi_senty_wen/一.png"
print(a.split(","))
print(len(a.split(",")))

# import pandas as pd

# df = pd.read_csv("../data/datasets_with_paths_fonts.csv", encoding="utf-8")
# print(f"Total rows: {len(df)}")
# print(f"Columns: {df.columns.tolist()}")

# # 检查image_paths列（假设为第5列，索引4）
# for idx, row in df.iterrows():
#     image_paths = row[4]  # 第5列
#     if pd.isna(image_paths):
#         print(f"Row {idx}: {row[0]} - image_paths is NaN")
#     elif not isinstance(image_paths, str):
#         print(f"Row {idx}: {row[0]} - image_paths is {type(image_paths)}: {image_paths}")
#     else:
#         paths = image_paths.split(",")
#         if len(paths) != 5:
#             print(f"Row {idx}: {row[0]} - has {len(paths)} paths: {image_paths}")




import pandas as pd

# 读取 CSV
df = pd.read_csv("../data/datasets_with_paths_fonts.csv", encoding="utf-8")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# 检查 image_paths 列
nan_count = df["image_paths"].isna().sum()
print(f"Number of rows with NaN image_paths: {nan_count}")

# 打印 NaN 行的详细信息
nan_rows = df[df["image_paths"].isna()]
if not nan_rows.empty:
    print("Rows with NaN image_paths:")
    print(nan_rows[["汉字", "笔顺", "image_paths"]])

# 检查非字符串值
non_string_rows = df[~df["image_paths"].apply(lambda x: isinstance(x, str) and pd.notna(x))]
if not non_string_rows.empty:
    print("Rows with non-string image_paths:")
    print(non_string_rows[["汉字", "笔顺", "image_paths"]])

# 保存 NaN 汉字到文件
if nan_count > 0:
    nan_chars = nan_rows["汉字"].tolist()
    with open("nan_chars.txt", "w", encoding="utf-8") as f:
        for char in nan_chars:
            f.write(f"{char}\n")
    print(f"NaN characters saved to nan_chars.txt")