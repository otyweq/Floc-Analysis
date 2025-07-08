import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image


def calculate_features_and_threshold(image):
    # 转换图像到灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用Otsu的方法自动确定阈值并应用
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 查找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    floc_count = len(contours)
    diameters = [np.sqrt(4 * cv2.contourArea(c) / np.pi) for c in contours]
    floc_density = np.sum(binary_image == 255) / binary_image.size
    return _, floc_count, np.mean(diameters) if diameters else 0, floc_density


def process_images_and_save_results(input_directory, results_file):
    results = []
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            image_path = os.path.join(input_directory, filename)
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)[:, :, ::-1].copy()  # PIL to OpenCV
            threshold, floc_count, avg_diameter, floc_density = calculate_features_and_threshold(image)
            results.append({'Filename': filename, 'Threshold': threshold,
                            'Floc Count': floc_count, 'Average Diameter': avg_diameter,
                            'Floc Density': floc_density})

    # 创建DataFrame并保存到Excel
    df = pd.DataFrame(results)
    df.to_excel(results_file, index=False)
    print("Completed processing images. Results saved to", results_file)


# 设置路径
input_dir = '采集图像/第一组转换'  # 图片目录
results_file = '采集图像/第一组处理/result.xlsx'  # 结果Excel文件


process_images_and_save_results(input_dir, results_file)
