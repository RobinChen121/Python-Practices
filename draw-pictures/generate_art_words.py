"""
created on 2025/2/10, 18:06
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""
from PIL import Image, ImageDraw, ImageFont
import imageio
import os

# 确定字体路径（macOS 默认字体）
font_path = "/System/Library/Fonts/PingFang.ttc"  # macOS自带字体

# 检查字体是否存在
if not os.path.exists(font_path):
    raise FileNotFoundError(f"字体文件未找到: {font_path}")

# 创建 GIF 帧
frames = []
text = "Python\n 数据科学"
font_size = 50

# 生成 10 帧动画
for i in range(50):
    img = Image.new("RGB", (300, 200), (255, 255, 255))  # 创建白色背景图片
    draw = ImageDraw.Draw(img)

    # 加载字体
    font = ImageFont.truetype(font_path, font_size)

    # 设置渐变颜色
    color = (255 - i * 20, 0, i * 20)  # 颜色变化（红-紫）

    # 绘制文字
    draw.text((50, 25), text, font=font, fill=color)

    frames.append(img)  # 添加帧

# 生成 GIF
output_path = "chinese_text.gif"
frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

print(f"GIF 生成完成，保存为 {output_path}")
