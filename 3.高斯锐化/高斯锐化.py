import numpy as np
from scipy.ndimage import convolve
import cv2

def gaussian_sharpen(image, sigma, strength):
    # 创建高斯核
    size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2)/(2*sigma**2)), (size, size))
    kernel /= np.sum(kernel)  # 归一化

    # 对图像进行卷积操作
    blurred = convolve(image, kernel)

    # 高斯锐化：原始图像加上锐化图像
    sharpened = image + (image - blurred) * strength

    # 将像素值限制在0到255之间
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened

# 读取图像
image = cv2.imread('img.jpg', 0)  # 以灰度模式读取图像

# 调用高斯锐化函数
sharpened_image = gaussian_sharpen(image, 2.0, 1.5)

# 显示原始图像和锐化后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图像
cv2.imwrite('original_image.jpg', image)
cv2.imwrite('sharpened_image.jpg', sharpened_image)