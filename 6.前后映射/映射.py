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

# 向前映射函数
def forward_mapping_rotation(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 进行前向映射
    rotated_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            # 计算旋转后的坐标
            new_coords = np.dot(rotation_matrix, [x, y, 1])
            new_x, new_y = new_coords[:2]
            
            # 边界检查
            if 0 <= new_x < width and 0 <= new_y < height:
                # 将像素值赋给旋转后的位置
                rotated_image[int(new_y), int(new_x),:] = image[y, x, :]
    
    return rotated_image

# 向后映射函数
def inverse_mapping_rotation(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    
    # 计算反向旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # 进行反向映射
    rotated_image = np.zeros_like(image)
    
    for y in range(height):
        for x in range(width):
            # 计算反向旋转后的坐标
            original_coords = np.dot(rotation_matrix, [x, y, 1])
            original_x, original_y = original_coords[:2]
            
            # 边界检查
            if 0 <= original_x < width and 0 <= original_y < height:
                # 将像素值赋给旋转前的位置
                rotated_image[y, x] = image[int(original_y), int(original_x)]
    
    return rotated_image

# 读取图像
image = cv2.imread('..\img.jpg')

# 向前映射旋转
forward_rotated_image = forward_mapping_rotation(image, 30)

# 向后映射旋转
inverse_rotated_image = inverse_mapping_rotation(image, 30)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Forward Rotated Image', forward_rotated_image)
cv2.imshow('Inverse Rotated Image', inverse_rotated_image)
# cv2.imwrite('forward_rotated_image.jpg', forward_rotated_image)
# cv2.imwrite('inverse_rotated_image.jpg', inverse_rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()