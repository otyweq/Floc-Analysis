import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image


# def convert_images_to_png(input_directory, output_directory):
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#     for filename in os.listdir(input_directory):
#         if filename.lower().endswith('.tif'):
#             input_path = os.path.join(input_directory, filename)
#             output_filename = f"{os.path.splitext(filename)[0]}.png"
#             output_path = os.path.join(output_directory, output_filename)
#             image = Image.open(input_path)
#             image.save(output_path)
#     print("Image conversion completed.")

def calculate_features(image_path, threshold):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    floc_count = len(contours)
    diameters = [np.sqrt(4 * cv2.contourArea(c) / np.pi) for c in contours]
    floc_density = np.sum(binary_image == 255) / binary_image.size
    return floc_count, np.mean(diameters) if diameters else 0, floc_density, binary_image

def process_images(input_directory, output_directory, threshold):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for filename in os.listdir(input_directory):
        if filename.lower().endswith('.png'):
            path = os.path.join(input_directory, filename)
            floc_count, avg_diameter, floc_density, processed_image = calculate_features(path, threshold)
            print(f"{filename}: Floc Count = {floc_count}, Avg Diameter = {avg_diameter:.2f}, Floc Density = {floc_density:.4f}")
            cv2.imwrite(os.path.join(output_directory, filename), processed_image)
    print("Image processing completed.")


def process_images_and_save_results(input_directory, output_directory, threshold, results_file):
    # 创建一个空的DataFrame
    results_df = pd.DataFrame(columns=['Filename', 'Floc Count', 'Average Diameter', 'Floc Density'])

    for filename in os.listdir(input_directory):
        if filename.lower().endswith('.png'):
            path = os.path.join(input_directory, filename)
            floc_count, avg_diameter, floc_density, processed_image = calculate_features(path, threshold)

            # 将结果添加到DataFrame
            results_df = results_df.append({
                'Filename': filename,
                'Floc Count': floc_count,
                'Average Diameter': avg_diameter,
                'Floc Density': floc_density
            }, ignore_index=True)

            # 保存处理后的图像
            cv2.imwrite(os.path.join(output_directory, filename), processed_image)

    # 保存结果到Excel文件
    results_df.to_excel(results_file, index=False)
    print("Image processing and result saving completed.")

# Set your directories here
input_dir = '采集图像/第1组057'
converted_dir = r'采集图像/第一组转换'
processed_dir = r'采集图像/第一组处理'
results_file = r'采集图像/第一组转换'

# # Convert TIF to PNG
# convert_images_to_png(input_dir, converted_dir)

# Determine an initial threshold using Otsu's method (You may choose any image from converted_dir)
sample_image_path = os.path.join(converted_dir, os.listdir(converted_dir)[0])
sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(sample_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 处理图像并保存结果
process_images_and_save_results(converted_dir, processed_dir, thresh, results_file)