import csv

def find_missing_indices(csv_file_path):
    third_column_raw = []  # 存储第三列原始值
    third_column_nums = []  # 存储转换后的数字
    actual_line_count = 0  # 手动计数行数
    
    # 检查文件总行数
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for line in file)
    print(f"文件总行数: {total_lines}")
    
    # 读取CSV文件
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader, None)  # 读取标题行
            if header:
                actual_line_count += 1
                print(f"标题行: {header}")
                third_column_raw.append(header[2] if len(header) >= 3 else "无第三列")
            
            # 读取每一行
            for row in csv_reader:
                actual_line_count += 1
                # 无论是否有第三列，都记录
                if len(row) >= 3:
                    value = row[2].strip()  # 去除空格
                    third_column_raw.append(value)
                    try:
                        num = int(value)
                        third_column_nums.append(num)
                    except ValueError:
                        print(f"第 {actual_line_count} 行非数字值: '{value}'")
                else:
                    third_column_raw.append("无第三列")
                    print(f"第 {actual_line_count} 行缺少第三列")
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_file_path}")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return

    # 打印所有第三列原始值
    print("\n读取到的第三列原始值（所有行）：")
    for i, val in enumerate(third_column_raw, start=1):
        print(f"行 {i}: '{val}'")
    
    # 打印成功转换的数字
    print("\n成功转换为数字的第三列值：")
    for i, num in enumerate(third_column_nums, start=1):
        print(f"行 {i}: {num}")
    
    if not third_column_nums:
        print("错误：没有找到有效的数字数据")
        return

    # 检查缺失的数字
    expected_range = set(range(1, 8106))
    actual_numbers = set(third_column_nums)
    missing_numbers = sorted(expected_range - actual_numbers)
    
    # 输出结果
    print(f"\n总共找到 {len(third_column_nums)} 个有效数字")
    print(f"总共处理 {actual_line_count} 行")
    if missing_numbers:
        print(f"总共缺失 {len(missing_numbers)} 个数字")
        print("缺失的下标如下：")
        print(missing_numbers)
    else:
        print("没有缺失任何数字！")
    
    # 检查重复数字
    if len(third_column_nums) != len(actual_numbers):
        seen = set()
        duplicates = set()
        for num in third_column_nums:
            if num in seen:
                duplicates.add(num)
            seen.add(num)
        print(f"警告：发现重复的数字：{sorted(duplicates)}")

# 使用示例
if __name__ == "__main__":
    file_path = "/Users/zhengshuyan/Downloads/GithubCode/extracted_data_3.csv"
    find_missing_indices(file_path)