import fitz  # pymupdf
import pandas as pd
import re

# 指定 PDF 文件路径
pdf_path = "/Users/zhengshuyan/Downloads/笔顺/3/575-580.pdf"
doc = fitz.open(pdf_path)


all_tables = []

# 遍历 PDF 的每一页
for page_num in range(len(doc)):
    page = doc[page_num]
    tables = page.find_tables()  # 检测页面中的表格

    print(f"第 {page_num} 页检测到的表格数量：{len(tables.tables)}")

    for table_idx in range(min(2, len(tables.tables))):  # 限制只处理前两个表格
        table_df = tables.tables[table_idx].to_pandas()

        print(f"第 {page_num} 页表格 {table_idx} 原始数据：")
        print(table_df)

        if table_df.empty:
            print(f"第 {page_num} 页表格 {table_idx} 为空，跳过处理")
            continue
        if table_idx == 1 and len(table_df) > 0 and table_df.iloc[0, 0] == "汉字":
            table_df = table_df[1:].reset_index(drop=True)

        expected_columns = ["汉字", "笔顺", "《字表》\n序号", "UCS"]
        if len(table_df.columns) >= len(expected_columns):
            table_df = table_df.iloc[:, :4]  # 确保只取前4列
            table_df.columns = expected_columns
            print(f"第 {page_num} 页表格 {table_idx} 设置列名后数据：")
            print(table_df)
        else:
            print(f"第 {page_num} 页表格 {table_idx} 列数不足，跳过处理")
            continue

        table_df["笔顺"] = table_df["笔顺"].str.replace("\n", " ", regex=False)
        table_df["笔顺"] = table_df["笔顺"].str.replace(r"[^0-9\s]", "", regex=True)
        table_df["笔顺"] = (
            table_df["笔顺"].str.replace(r"\s+", " ", regex=True).str.strip()
        )
        print(f"第 {page_num} 页表格 {table_idx} 清理笔顺后数据：")
        print(table_df.head())

        # 清理 'UCS' 列
        def clean_ucs(value):
            if pd.isna(value):
                print(f"UCS 值为空: {value}")
                return None
            value = str(value).replace("\n", "")
            match = re.search(r"[0-9A-F]{4,}", value)
            if match:
                return match.group(0)
            else:
                print(f"UCS 值无效: {value}")
                return value

        table_df["UCS"] = table_df["UCS"].apply(clean_ucs)
        print(f"第 {page_num} 页表格 {table_idx} 清理 UCS 后数据：")
        print(table_df.head())

        all_tables.append(table_df)

print(f"all_tables 包含 {len(all_tables)} 个表格")

if all_tables:
    final_df = pd.concat(all_tables, ignore_index=True)
else:
    final_df = pd.DataFrame()


final_df.to_csv("test.csv", index=False, encoding="utf-8-sig")

doc.close()

print("提取的数据预览：")
print(final_df)
