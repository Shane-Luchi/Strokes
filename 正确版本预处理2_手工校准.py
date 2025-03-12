import fitz  # pymupdf
import pandas as pd
import re

# 指定 PDF 文件路径
pdf_path = "/Users/zhengshuyan/Downloads/笔顺/1/009-581.pdf"
doc = fitz.open(pdf_path)

all_tables = []

# 遍历 PDF 的每一页
for page_num in range(len(doc)):
    page = doc[page_num]
    tables = page.find_tables()  # 检测页面中的表格

    for table_idx in range(min(2, len(tables.tables))):  # 限制只处理前两个表格
        table_df = tables.tables[table_idx].to_pandas()

        if table_df.empty:
            continue
        if table_idx == 1 and len(table_df) > 0 and table_df.iloc[0, 0] == "汉字":
            table_df = table_df[1:].reset_index(drop=True)

        expected_columns = ["汉字", "笔顺", "《字表》\n序号", "UCS"]
        if len(table_df.columns) >= len(expected_columns):
            table_df = table_df.iloc[:, :4]  # 确保只取前4列
            table_df.columns = expected_columns
        else:
            continue

        # 第二列（笔顺）处理
        table_df["笔顺"] = table_df["笔顺"].str.replace("\n", " ", regex=False)
        table_df["笔顺"] = table_df["笔顺"].str.replace(r"[^0-9\s]", "", regex=True)
        table_df["笔顺"] = (
            table_df["笔顺"].str.replace(r"\s+", " ", regex=True).str.strip()
        )

        # 第三列（《字表》\n序号）：删除所有换行符
        table_df["《字表》\n序号"] = table_df["《字表》\n序号"].str.replace("\n", "", regex=False)

        # 第四列（UCS）清理
        def clean_ucs(value):
            if pd.isna(value):
                return None
            value = str(value).replace("\n", "")
            match = re.search(r"[0-9A-F]{4,}", value)
            return match.group(0) if match else value

        table_df["UCS"] = table_df["UCS"].apply(clean_ucs)

        # 打印每个表格处理后的最终结果
        print(f"第 {page_num} 页表格 {table_idx} 处理后的数据：")
        print(table_df)
        
        all_tables.append(table_df)

if all_tables:
    final_df = pd.concat(all_tables, ignore_index=True)
else:
    final_df = pd.DataFrame()

final_df.to_csv("test.csv", index=False, encoding="utf-8-sig")

doc.close()

print("提取的数据预览：")
print(final_df)