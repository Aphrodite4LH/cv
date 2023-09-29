import numpy as np
import cv2


def gaussian_blur(image_path, kernel_size, sigma):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

    # 创建高斯核
    kernel = create_gaussian_kernel(kernel_size, sigma)

    # 执行滤波操作
    blurred_image = convolve(image, kernel)

    # 返回滤波后的图像
    return blurred_image


def create_gaussian_kernel(kernel_size, sigma):
    # 计算高斯核的大小
    size = int((kernel_size - 1) / 2)

    # 创建高斯核
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - size) ** 2 + (y - size) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size))

    # 归一化高斯核
    kernel /= np.sum(kernel)

    return kernel


def convolve(image, kernel):
    # 获取图像的尺寸
    height, width = image.shape

    # 获取高斯核的尺寸
    kernel_size = kernel.shape[0]

    # 计算高斯核的大小
    size = int((kernel_size - 1) / 2)

    # 创建一个新的图像对象（空白），用于存储滤波后的图像
    blurred_image = np.zeros((height, width), dtype=np.uint8) # uint8:无符号八位整数，作为数组的数据类型

    # 执行卷积操作（最边上的像素点不进行卷积）
    for row in range(size, height - size):
        for col in range(size, width - size):
            # 提取当前像素周围的图像块
            block = image[row - size: row + size + 1, col - size: col + size + 1]

            # 进行卷积计算
            convolved_value = np.sum(block * kernel)

            # 将卷积结果存储到新图像中
            blurred_image[row, col] = int(convolved_value)

    return blurred_image


# 使用示例
input_image_path = "img.jpg"  # 输入图像路径
output_image_path = "gaussian100.jpg"  # 输出图像路径
kernel_size = 5  # 高斯核大小
sigma = 100.0  # 高斯核标准差

blurred_image = gaussian_blur(input_image_path, kernel_size, sigma)
cv2.imwrite(output_image_path, blurred_image)
