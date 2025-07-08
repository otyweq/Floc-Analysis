import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
import logging

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image_with_pillow_convert_to_opencv(image_path: str) -> np.ndarray:
    """使用PIL加载图像并转换为OpenCV兼容格式。"""
    try:
        pil_image = Image.open(image_path).convert('RGB')
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # 将RGB转换为BGR
        return open_cv_image
    except Exception as e:
        logging.error(f"加载图像 {image_path} 失败: {e}")
        return None

def calculate_threshold_and_features(image: np.ndarray):
    """使用OpenCV计算图像阈值和几何特征。"""
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        floc_count = len(contours)
        diameters = [np.sqrt(4 * cv2.contourArea(c) / np.pi) for c in contours]
        floc_density = np.sum(binary_image == 255) / binary_image.size
        return threshold, floc_count, np.mean(diameters) if diameters else 0, floc_density
    except Exception as e:
        logging.error(f"计算特征错误: {e}")
        return None, None, None, None

def process_images_and_save_results(input_directory: str, results_file: str):
    """处理目录中所有图像并将结果保存到Excel文件中。"""
    results = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    try:
        for filename in os.listdir(input_directory):
            if filename.lower().endswith(tuple(valid_extensions)):
                image_path = os.path.join(input_directory, filename)
                image = load_image_with_pillow_convert_to_opencv(image_path)
                if image is not None:
                    threshold, floc_count, avg_diameter, floc_density = calculate_threshold_and_features(image)
                    results.append({
                        'Filename': filename,
                        'Threshold': threshold,
                        'Floc Count': floc_count,
                        'Average Diameter': avg_diameter,
                        'Floc Density': floc_density
                    })
        df = pd.DataFrame(results)
        df.to_excel(results_file, index=False, engine='openpyxl')
        logging.info(f"图像处理完成。结果已保存到 {results_file}")
    except Exception as e:
        logging.error(f"处理图像失败: {e}")

# 设置路径
input_dir = '采集图像/第一组转换'
results_file = '采集图像/第一组处理/result3.xlsx'

# 执行图像处理
process_images_and_save_results(input_dir, results_file)
