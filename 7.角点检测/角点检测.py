import os
import cv2
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 获取当前脚本文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# 将当前工作目录设置为代码文件所在的目录
os.chdir(current_dir)


def harris_corner_detection(image, threshold=0.01):
    # 1. 计算图像的梯度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 2. 计算 Harris 角点响应函数
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy

    # 计算局部自相关矩阵的和
    window_size = 3
    kernel = np.ones((window_size, window_size), np.float32)
    dx2_sum = cv2.filter2D(dx2, -1, kernel)
    dy2_sum = cv2.filter2D(dy2, -1, kernel)
    dxy_sum = cv2.filter2D(dxy, -1, kernel)

    # 计算 Harris 响应函数
    det = dx2_sum * dy2_sum - dxy_sum * dxy_sum
    trace = dx2_sum + dy2_sum
    harris_response = det - 0.04 * (trace ** 2)

    # 3. 应用阈值，筛选出角点
    corners = np.argwhere(harris_response > threshold * harris_response.max())
    corners = corners[:, ::-1]  # 转换为 (x, y) 坐标格式

    return corners

# 读取图像
image = cv2.imread('../img.jpg')

# 进行 Harris 角点检测
corners = harris_corner_detection(image)

# 在图像上绘制角点
for corner in corners:
    x, y = corner
    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

# 显示结果图像
cv2.imshow('Harris Corner Detection', image)
cv2.imwrite('1.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()