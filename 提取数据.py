import fitz  # pymupdf
import pandas as pd
import re

# 指定 PDF 文件路径
pdf_path = "/Users/zhengshuyan/Downloads/笔顺/3/575-580.pdf"
doc = fitz.open(pdf_path)
print("PDF 元数据：", doc.metadata)

all_tables = []

# 遍历 PDF 的每一页
for page_num in range(len(doc)):
    page = doc[page_num]
    tables = page.find_tables()  # 检测页面中的表格

    # 打印页面原始文本以调试
    raw_text = page.get_text()
    print(f"第 {page_num} 页原始文本：\n{raw_text}\n")

    for table_idx in range(min(2, len(tables.tables))):  # 限制只处理前两个表格
        table_df = tables.tables[table_idx].to_pandas()

        if table_df.empty:
            print(f"第 {page_num} 页表格 {table_idx} 为空，跳过")
            continue
        if table_idx == 1 and len(table_df) > 0 and table_df.iloc[0, 0] == "汉字":
            table_df = table_df[1:].reset_index(drop=True)

        expected_columns = ["汉字", "笔顺", "《字表》\n序号", "UCS"]
        if len(table_df.columns) >= len(expected_columns):
            table_df = table_df.iloc[:, :4]  # 确保只取前4列
            table_df.columns = expected_columns
        else:
            print(f"第 {page_num} 页表格 {table_idx} 列数不足，跳过")
            continue

        # 修复汉字乱码，优先尝试 UTF-16（UCS 兼容）
        def fix_encoding(text):
            if pd.isna(text):
                return text
            original_text = text
            try:
                # 优先尝试 UTF-16
                decoded_text = text.encode().decode('utf-16', errors='replace')
                if any(ord(c) > 127 for c in decoded_text):  # 确认包含中文
                    print(f"成功使用 UTF-16 解码：{original_text} -> {decoded_text}")
                    return decoded_text
                # 备用 UTF-8
                decoded_text = text.encode().decode('utf-8', errors='replace')
                print(f"使用 UTF-8 解码：{original_text} -> {decoded_text}")
                return decoded_text
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                print(f"解码失败，原始值：{original_text}, 错误：{e}")
                return original_text

        table_df["汉字"] = table_df["汉字"].apply(fix_encoding)

        # 后续处理逻辑保持不变...

        # 打印每个表格处理后的最终结果
        print(f"第 {page_num} 页表格 {table_idx} 处理后的数据：")
        print(table_df)
        
        all_tables.append(table_df)

if all_tables:
    final_df = pd.concat(all_tables, ignore_index=True)
else:
    final_df = pd.DataFrame()

final_df.to_csv("test3.csv", index=False, encoding="utf-8-sig")

doc.close()

print("提取的数据预览：")
print(final_df)