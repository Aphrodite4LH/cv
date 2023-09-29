import os
import cv2
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 获取当前脚本文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# 将当前工作目录设置为代码文件所在的目录
os.chdir(current_dir)

# 读取输入图像
input_image = cv2.imread('..\img.jpg')

# 定义放大比例
scale_percent = 2  # 放大比例为2倍

# 最近邻插值
nearest_neighbor = cv2.resize(input_image, None, fx=scale_percent, fy=scale_percent, interpolation=cv2.INTER_NEAREST)

# 双线性插值
bilinear = cv2.resize(input_image, None, fx=scale_percent, fy=scale_percent, interpolation=cv2.INTER_LINEAR)

# 三次样条插值
bicubic = cv2.resize(input_image, None, fx=scale_percent, fy=scale_percent, interpolation=cv2.INTER_CUBIC)

# 保存放大后的图像到本地
cv2.imwrite('nearest_neighbor.jpg', nearest_neighbor)
cv2.imwrite('bilinear.jpg', bilinear)
cv2.imwrite('bicubic.jpg', bicubic)