import os
from PIL import Image, ImageOps
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd

# 设置全局字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def weighted_gray(image):
    gray = ImageOps.grayscale(image)
    return gray

def extract_floc_diameter(thresh_image_array):
    contours, _ = cv2.findContours(thresh_image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diameters = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0:
            diameter = 2 * np.sqrt(area / np.pi)
            diameters.append(diameter)
    return diameters

def calculate_floc_density(thresh_image_array):
    floc_pixels = np.sum(thresh_image_array == 255)
    total_pixels = thresh_image_array.size
    density = floc_pixels / total_pixels
    return density

def calculate_fractal_dimension(thresh_image_array, plot_path):
    contours, _ = cv2.findContours(thresh_image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    log_areas = []
    log_perimeters = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > 0 and perimeter > 0:
            log_areas.append(math.log(area))
            log_perimeters.append(math.log(perimeter))

    if not log_areas or not log_perimeters:
        return None

    coeffs = np.polyfit(log_perimeters, log_areas, 1)
    fractal_dimension = coeffs[0]

    plt.figure()
    plt.plot(log_perimeters, log_areas, 'o', label='数据点')
    plt.plot(log_perimeters, np.polyval(coeffs, log_perimeters), label='拟合线')
    plt.xlabel('周长的对数值')
    plt.ylabel('面积的对数值')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

    return fractal_dimension

def extract_floc_count(thresh_image_array):
    contours, _ = cv2.findContours(thresh_image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def process_images(input_folder, output_folder_gray, output_folder_thresh, output_folder_results, output_folder_plots):
    results = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.bmp'):
            image_path = os.path.join(input_folder, filename)
            if not os.path.exists(image_path):
                continue

            try:
                with Image.open(image_path) as img:
                    gray_image = weighted_gray(img)
                    gray_image_array = np.array(gray_image)
                    _, thresh_image_array = cv2.threshold(gray_image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    thresh_image = Image.fromarray(thresh_image_array)

                    floc_count = extract_floc_count(thresh_image_array)
                    diameters = extract_floc_diameter(thresh_image_array)
                    avg_diameter = np.mean(diameters) if diameters else 0
                    floc_density = calculate_floc_density(thresh_image_array)

                    plot_path = os.path.join(output_folder_plots, f'fractal_{filename.replace(".bmp", ".png")}')
                    fractal_dimension = calculate_fractal_dimension(thresh_image_array, plot_path)

                    print(f'文件: {filename}, 絮体数量: {floc_count}, 平均粒径: {avg_diameter}, 密度: {floc_density}, 分形维数: {fractal_dimension}')

                    gray_image_path = os.path.join(output_folder_gray, f'gray_{filename.replace(".bmp", ".png")}')
                    thresh_image_path = os.path.join(output_folder_thresh, f'thresh_{filename.replace(".bmp", ".png")}')

                    gray_image.save(gray_image_path, 'PNG')
                    thresh_image.save(thresh_image_path, 'PNG')

                    results.append({
                        '文件': filename,
                        '絮体数量': floc_count,
                        '平均粒径': avg_diameter,
                        '密度': floc_density,
                        '分形维数': fractal_dimension
                    })

            except Exception as e:
                print(f"处理失败 {image_path}: {e}")
                continue

    df_results = pd.DataFrame(results)
    result_csv_path = os.path.join(output_folder_results, '分析结果.csv')
    df_results.to_csv(result_csv_path, index=False, encoding='utf-8-sig')

    print('批量处理完成，结果已保存到表格中。')
