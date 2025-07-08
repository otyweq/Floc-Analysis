import cv2
import numpy as np
import os
from skimage.measure import label, regionprops


def process_image(image_path):
    # 读取和预处理图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 寻找轮廓来估计絮体数量
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    floc_count = len(contours)

    # 计算絮体粒径和图像密度
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    image_area = binary_image.shape[0] * binary_image.shape[1]
    floc_density = total_area / image_area if image_area else 0

    # 计算平均粒径
    average_diameter = np.mean(
        [2 * np.sqrt(cv2.contourArea(contour) / np.pi) for contour in contours]) if contours else 0

    # 分形维数简化计算（基于面积和周长）
    fractal_dimensions = [np.log(cv2.contourArea(contour)) / np.log(cv2.arcLength(contour, True)) for contour in
                          contours if cv2.contourArea(contour) > 0 and cv2.arcLength(contour, True) > 0]
    average_fractal_dimension = np.mean(fractal_dimensions) if fractal_dimensions else 0

    return {
        "floc_count": floc_count,
        "average_diameter": average_diameter,
        "floc_density": floc_density,
        "average_fractal_dimension": average_fractal_dimension
    }


def batch_process_images(directory_path):
    results = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            results[filename] = process_image(image_path)

    return results


# 图像目录路径
directory_path = 'path_to_your_directory'
results = batch_process_images(directory_path)

# 打印或以其他方式使用结果
for filename, result in results.items():
    print(f"Image: {filename}")
    print(f"  Floc Count: {result['floc_count']}")
    print(f"  Average Diameter: {result['average_diameter']:.2f}")
    print(f"  Floc Density: {result['floc_density']:.4f}")
    print(f"  Average Fractal Dimension: {result['average_fractal_dimension']:.4f}\n")
