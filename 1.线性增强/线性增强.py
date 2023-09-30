from PIL import Image


def linear_enhancement(image_path, alpha, beta):
    # 打开图像
    image = Image.open(image_path)

    # 获取图像的宽度和高度
    width, height = image.size

    # 创建一个新的图像对象，用于存储增强后的图像
    enhanced_image = Image.new("RGB", (width, height))

    # 遍历图像的每个像素点，并进行线性增强
    for x in range(width):
        for y in range(height):
            # 获取原始像素值
            r, g, b = image.getpixel((x, y))

            # 进行线性增强计算
            r_enhanced = int(alpha * r + beta)
            g_enhanced = int(alpha * g + beta)
            b_enhanced = int(alpha * b + beta)

            # 将增强后的像素值设置到新图像中
            enhanced_image.putpixel((x, y), (r_enhanced, g_enhanced, b_enhanced))

    # 返回增强后的图像
    return enhanced_image


# 使用示例
input_image_path = "图片样例.jpg"  # 输入图像路径
output_image_path = "线性增强后3.jpg"  # 输出图像路径
alpha = 0.8  # 增强系数
beta = 30  # 增强偏移量

enhanced_image = linear_enhancement(input_image_path, alpha, beta)
enhanced_image.save(output_image_path)
