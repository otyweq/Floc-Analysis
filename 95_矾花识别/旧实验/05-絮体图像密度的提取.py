import cv2
import numpy as np


def calculate_floc_image_density(binary_image):
    # 计算二值图像中的白色像素点数（絮体像素）
    white_pixels = np.sum(binary_image == 255)
    # 计算图像的总像素数
    total_pixels = binary_image.size
    # 计算絮体图像密度
    floc_density = white_pixels / total_pixels
    return floc_density


def process_and_calculate_density(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用阈值分割
    _, binary_image = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)

    # 计算絮体图像密度
    density = calculate_floc_image_density(binary_image)

    # 打印絮体图像密度
    print(f"Floc Image Density: {density:.4f}")

    # 可视化处理后的二值图像（可选）
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 替换为您的图像路径
image_path = 'path_to_your_image.jpg'
process_and_calculate_density(image_path)
