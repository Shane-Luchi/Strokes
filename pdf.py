import fitz  # pymupdf
import pandas as pd
import re

# 指定 PDF 文件路径
pdf_path = "/Users/zhengshuyan/Downloads/笔顺/1/009-581.pdf"
try:
    doc = fitz.open(pdf_path)
except Exception as e:
    print(f"无法打开 PDF 文件：{str(e)}")
    exit(1)

# 初始化用于存储左右表格的列表
left_tables = []
right_tables = []

# 遍历 PDF 的每一页
for page_num in range(len(doc)):
    try:
        page = doc[page_num]
        tables = page.find_tables(strategy="lines", snap_tolerance=15)  # 优化表格检测

        print(f"第 {page_num} 页检测到的表格数量：{len(tables.tables)}")

        # 如果页面检测到的表格少于 2 个，记录警告
        if len(tables.tables) < 2:
            print(f"警告：第 {page_num} 页检测到的表格数量为 {len(tables.tables)}，少于预期 2 个")
            continue

        # 提取左侧和右侧表格
        left_table = tables.tables[0].to_pandas()
        right_table = tables.tables[1].to_pandas()

        # 打印原始数据以调试
        print(f"第 {page_num} 页 left_table 原始数据：")
        print(left_table.head() if not left_table.empty else "空表格")
        print(f"第 {page_num} 页 right_table 原始数据：")
        print(right_table.head() if not right_table.empty else "空表格")

        # 检查表格是否为空
        if left_table.empty:
            print(f"第 {page_num} 页 left_table 为空，跳过处理")
            left_table = None
        if right_table.empty:
            print(f"第 {page_num} 页 right_table 为空，跳过处理")
            right_table = None

        # 移除表头行（检查所有表格的第一行是否包含表头关键字）
        expected_columns = ["汉字", "笔顺", "《字表》\n序号", "UCS"]
        for table, table_name in [(left_table, "left_table"), (right_table, "right_table")]:
            if table is not None and len(table) > 0:
                first_row = table.iloc[0].astype(str)
                if any(keyword in first_row.values for keyword in expected_columns):
                    table = table[1:].reset_index(drop=True)
                    print(f"第 {page_num} 页 {table_name} 移除表头行")
                    print(f"移除表头后 {table_name} 数据：")
                    print(table.head() if not table.empty else "空表格")

        # 设置列名并调整列数
        if left_table is not None:
            if len(left_table.columns) != len(expected_columns):
                print(
                    f"第 {page_num} 页的 left_table 列数不匹配，原始列数: {len(left_table.columns)}, 原始列名: {left_table.columns.tolist()}"
                )
                if len(left_table.columns) >= 4:
                    left_table = left_table.iloc[:, :4]
                    left_table.columns = expected_columns
                    print(f"调整后 left_table 列名: {left_table.columns.tolist()}")
                else:
                    print(f"第 {page_num} 页的 left_table 列数不足，跳过处理")
                    left_table = None
            else:
                left_table.columns = expected_columns

        if right_table is not None:
            if len(right_table.columns) != len(expected_columns):
                print(
                    f"第 {page_num} 页的 right_table 列数不匹配，原始列数: {len(right_table.columns)}, 原始列名: {right_table.columns.tolist()}"
                )
                if len(right_table.columns) >= 4:
                    right_table = right_table.iloc[:, :4]
                    right_table.columns = expected_columns
                    print(f"调整后 right_table 列名: {right_table.columns.tolist()}")
                else:
                    print(f"第 {page_num} 页的 right_table 列数不足，跳过处理")
                    right_table = None
            else:
                right_table.columns = expected_columns

        # 打印处理后的表格信息
        if left_table is not None and not left_table.empty:
            print("left_table 的列名:", left_table.columns.tolist())
            print("完整 left_table 数据：")
            print(left_table)
        if right_table is not None and not right_table.empty:
            print("right_table 的列名:", right_table.columns.tolist())
            print("完整 right_table 数据：")
            print(right_table)

        # 检查是否有 "8096" 的行
        if left_table is not None and "8096" in left_table["《字表》\n序号"].values:
            print(f"第 {page_num} 页的 left_table 包含 '8096' 的行:")
            print(left_table[left_table["《字表》\n序号"] == "8096"])
        if right_table is not None and "8096" in right_table["《字表》\n序号"].values:
            print(f"第 {page_num} 页的 right_table 包含 '8096' 的行:")
            print(right_table[right_table["《字表》\n序号"] == "8096"])

        # 处理数据
        if left_table is not None and not left_table.empty:
            # 从 '汉字' 列中提取第一个字符
            left_table["汉字"] = left_table["汉字"].str.split("\n").str[0]
            # 清理 '笔顺' 列
            left_table["笔顺"] = left_table["笔顺"].str.replace("\n", " ", regex=False)
            left_table["笔顺"] = left_table["笔顺"].str.replace(r"[^0-9\s]", "", regex=True)
            left_table["笔顺"] = (
                left_table["笔顺"].str.replace(r"\s+", " ", regex=True).str.strip()
            )
            # 清理 'UCS' 列
            left_table["UCS"] = left_table["UCS"].apply(
                lambda x: str(x).replace("\n", "") if pd.notna(x) else x
            )
            match = left_table["UCS"].str.match(r"[0-9A-F]{4,}", na=False)
            left_table.loc[~match, "UCS"] = left_table.loc[~match, "UCS"].apply(
                lambda x: re.search(r"[0-9A-F]{4,}", str(x)).group(0)
                if re.search(r"[0-9A-F]{4,}", str(x))
                else x
            )
            print("清理后的完整 left_table 数据：")
            print(left_table)
            left_tables.append(left_table)

        if right_table is not None and not right_table.empty:
            # 从 '汉字' 列中提取第一个字符
            right_table["汉字"] = right_table["汉字"].str.split("\n").str[0]
            # 清理 '笔顺' 列
            right_table["笔顺"] = right_table["笔顺"].str.replace("\n", " ", regex=False)
            right_table["笔顺"] = right_table["笔顺"].str.replace(
                r"[^0-9\s]", "", regex=True
            )
            right_table["笔顺"] = (
                right_table["笔顺"].str.replace(r"\s+", " ", regex=True).str.strip()
            )
            # 清理 'UCS' 列
            right_table["UCS"] = right_table["UCS"].apply(
                lambda x: str(x).replace("\n", "") if pd.notna(x) else x
            )
            match = right_table["UCS"].str.match(r"[0-9A-F]{4,}", na=False)
            right_table.loc[~match, "UCS"] = right_table.loc[~match, "UCS"].apply(
                lambda x: re.search(r"[0-9A-F]{4,}", str(x)).group(0)
                if re.search(r"[0-9A-F]{4,}", str(x))
                else x
            )
            print("清理后的完整 right_table 数据：")
            print(right_table)
            right_tables.append(right_table)
    except Exception as e:
        print(f"第 {page_num} 页处理时发生错误：{str(e)}")
        continue  # 跳过当前页面，继续处理下一页

# 打印合并前的表格数量
print(f"left_tables 包含 {len(left_tables)} 个表格")
print(f"right_tables 包含 {len(right_tables)} 个表格")

# 合并所有左侧和右侧表格
if left_tables:
    df_left = pd.concat(left_tables, ignore_index=True)
else:
    df_left = pd.DataFrame()

if right_tables:
    df_right = pd.concat(right_tables, ignore_index=True)
else:
    df_right = pd.DataFrame()

# 将左右表格合并为一个最终的 DataFrame
final_df = pd.concat([df_left, df_right], ignore_index=True)

# 将结果保存为 CSV 文件
final_df.to_csv("extracted_data_2.csv", index=False, encoding="utf-8-sig")

# 关闭 PDF 文件
doc.close()

# 输出最终数据以供检查
print("提取的数据预览：")
print(final_df)