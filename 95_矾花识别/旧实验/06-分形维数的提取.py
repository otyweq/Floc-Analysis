import cv2
import numpy as np


def calculate_fractal_dimension(contour):
    # 计算轮廓区域的面积
    area = cv2.contourArea(contour)
    # 计算轮廓的周长
    perimeter = cv2.arcLength(contour, True)
    if area == 0 or perimeter == 0:
        return 0
    # 使用面积和周长计算分形维数的简化方法
    fractal_dimension = np.log(area) / np.log(perimeter)
    return fractal_dimension


def process_and_calculate_fractal_dimensions(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用阈值分割
    _, binary_image = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)
    # 寻找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 对每个絮体轮廓计算分形维数
    fractal_dimensions = [calculate_fractal_dimension(contour) for contour in contours]

    # 打印分形维数结果
    for i, fractal_dimension in enumerate(fractal_dimensions, start=1):
        print(f"Floc {i}: Fractal Dimension = {fractal_dimension:.4f}")

    # 可视化轮廓（可选）
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Floc Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 替换为您的图像路径
image_path = 'path_to_your_image.jpg'
process_and_calculate_fractal_dimensions(image_path)
