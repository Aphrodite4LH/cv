# import os
# import cv2
# import numpy as np
# import sys
# sys.stdout.reconfigure(encoding='utf-8')

# # 获取当前脚本文件的路径
# current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)

# # 将当前工作目录设置为代码文件所在的目录
# os.chdir(current_dir)

# # 读取图像
# image = cv2.imread('..\img.jpg')

# # 平移变换
# translation_matrix = np.float32([[1, 0, 100], [0, 1, 50]])  # 平移变换矩阵
# translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

# # 旋转变换
# rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 30, 1)  # 旋转变换矩阵
# rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

# # 相似变换
# similarity_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 45, 0.8)  # 相似变换矩阵
# similar_image = cv2.warpAffine(image, similarity_matrix, (image.shape[1], image.shape[0]))

# # 仿射变换
# pts1 = np.float32([[50, 50], [200, 50], [50, 200]])  # 原图像上的三个点
# pts2 = np.float32([[50, 100], [200, 50], [100, 250]])  # 目标图像上的对应三个点
# affine_matrix = cv2.getAffineTransform(pts1, pts2)  # 仿射变换矩阵
# affine_image = cv2.warpAffine(image, affine_matrix, (image.shape[1], image.shape[0]))

# # 投影变换
# pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])  # 原图像上的四个点
# pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])  # 目标图像上的对应四个点
# perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)  # 投影变换矩阵
# perspective_image = cv2.warpPerspective(image, perspective_matrix, (300, 300))

# # 将图像移动到中央位置
# center_x = image.shape[1] // 2
# center_y = image.shape[0] // 2
# translate_x = center_x - image.shape[1] // 2
# translate_y = center_y - image.shape[0] // 2
# translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])  # 平移变换矩阵
# translated_image = cv2.warpAffine(translated_image, translation_matrix, (image.shape[1], image.shape[0]))
# rotated_image = cv2.warpAffine(rotated_image, translation_matrix, (image.shape[1], image.shape[0]))
# similar_image = cv2.warpAffine(similar_image, translation_matrix, (image.shape[1], image.shape[0]))
# affine_image = cv2.warpAffine(affine_image, translation_matrix, (image.shape[1], image.shape[0]))
# perspective_image = cv2.warpAffine(perspective_image, translation_matrix, (image.shape[1], image.shape[0]))

# # 保存图像
# cv2.imwrite('translated_image.jpg', translated_image)
# cv2.imwrite('rotated_image.jpg', rotated_image)
# cv2.imwrite('similar_image.jpg', similar_image)
# cv2.imwrite('affine_image.jpg', affine_image)
# cv2.imwrite('perspective_image.jpg', perspective_image)

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

# 读取图像
image = cv2.imread('..\img.jpg')

# 平移变换
translation_matrix = np.float32([[1, 0, 100], [0, 1, 50]])  # 平移变换矩阵
translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

# 旋转变换
rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 30, 1)  # 旋转变换矩阵
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

# 相似变换
similarity_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 45, 0.8)  # 相似变换矩阵
similar_image = cv2.warpAffine(image, similarity_matrix, (image.shape[1], image.shape[0]))

# 仿射变换
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])  # 原图像上的三个点
pts2 = np.float32([[50, 100], [200, 50], [100, 250]])  # 目标图像上的对应三个点
affine_matrix = cv2.getAffineTransform(pts1, pts2)  # 仿射变换矩阵
affine_image = cv2.warpAffine(image, affine_matrix, (image.shape[1], image.shape[0]))

# 投影变换
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])  # 原图像上的四个点
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])  # 目标图像上的对应四个点
perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)  # 投影变换矩阵
perspective_image = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]))

# 将图像移动到中央位置
center_x = image.shape[1] // 2
center_y = image.shape[0] // 2
translate_x = center_x - image.shape[1] // 2
translate_y = center_y - image.shape[0] // 2

# 创建独立的平移变换矩阵
translation_matrix_translated = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
translation_matrix_rotated = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
translation_matrix_similar = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
translation_matrix_affine = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
translation_matrix_perspective = np.float32([[1, 0, translate_x], [0, 1, translate_y]])

# 应用平移变换
translated_image = cv2.warpAffine(translated_image, translation_matrix_translated, (image.shape[1], image.shape[0]))
rotated_image = cv2.warpAffine(rotated_image, translation_matrix_rotated, (image.shape[1], image.shape[0]))
similar_image = cv2.warpAffine(similar_image, translation_matrix_similar, (image.shape[1], image.shape[0]))
affine_image = cv2.warpAffine(affine_image, translation_matrix_affine, (image.shape[1], image.shape[0]))
perspective_image = cv2.warpAffine(perspective_image, translation_matrix_perspective, (image.shape[1], image.shape[0]))

# 保存图像
cv2.imwrite('translated_image.jpg', translated_image)
cv2.imwrite('rotated_image.jpg', rotated_image)
cv2.imwrite('similar_image.jpg', similar_image)
cv2.imwrite('affine_image.jpg', affine_image)
cv2.imwrite('perspective_image.jpg', perspective_image)