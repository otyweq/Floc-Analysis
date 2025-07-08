import cv2
import numpy as np
import math


def calculate_equivalent_diameter(contour):
    # 计算轮廓区域的面积
    area = cv2.contourArea(contour)
    # 计算等效粒径
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    return equivalent_diameter


def process_and_measure_flocs(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用阈值分割
    _, binary_image = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)
    # 寻找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 对每个絮体轮廓计算等效粒径
    diameters = [calculate_equivalent_diameter(contour) for contour in contours]

    # 打印等效粒径结果
    for i, diameter in enumerate(diameters, start=1):
        print(f"Floc {i}: Equivalent Diameter = {diameter:.2f} pixels")

    # 可视化轮廓（可选）
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Floc Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 替换为您的图像路径
image_path = 'path_to_your_image.jpg'
process_and_measure_flocs(image_path)
