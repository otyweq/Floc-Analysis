import cv2
import numpy as np


def process_and_count_flocs(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用阈值分割，这里假设阈值为170，根据您的实际图像可能需要调整
    _, binary_image = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)

    # 寻找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算检测到的轮廓数量，即絮体数量
    floc_count = len(contours)

    # 打印絮体数量
    print(f"Detected floc count: {floc_count}")

    # 可视化轮廓（可选）
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Floc Contours", image)
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 替换为您的图像路径
image_path = 'path_to_your_image.jpg'
process_and_count_flocs(image_path)
