import csv
from PIL import Image, ImageDraw, ImageFont

# 设置图片大小（单位：像素）
img_size = (200, 200)  # 宽度200像素，高度200像素

# 设置字体路径和大小
font_path = "NotoSerifCJKsc-Regular.otf"  # 替换为你的字体文件路径
font_size = 150  # 字体大小（单位：点），可根据需要调整
font = ImageFont.truetype(font_path, font_size)

# 设置颜色
bg_color = (255, 255, 255)  # 背景颜色：白色
text_color = (0, 0, 0)      # 文字颜色：黑色

step = 0
# 读取CSV文件并生成图片
with open('datasets.csv', 'r', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):

        # 打几个样片
        # if i % 100 != 0:
        #     continue
        
        character = row[0]  # 提取第一列的汉字

        # 创建新图片
        img = Image.new('RGB', img_size, bg_color)
        draw = ImageDraw.Draw(img)

        # 获取文字的边界框并计算宽度和高度
        bbox = font.getbbox(character)
        text_width = bbox[2] - bbox[0]  # 右边界 - 左边界
        text_height = bbox[3] - bbox[1] # 下边界 - 上边界

        # 计算文字居中位置
        x = (img_size[0] - text_width) / 2

        # 更精确的垂直居中：考虑边界框的顶部偏移
        y = (img_size[1] - text_height) / 2 - bbox[1]  # 加上 top 的偏移量

        # 在图片上绘制文字
        draw.text((x, y), character, font=font, fill=text_color)



        # 保存图片，文件名即为汉字本身
        img.save(f"./pic/{character}.png")

        # 可选：显示进度
        if (i + 1) % 100 == 0:
            print(f"已生成 {i + 1} 张图片")