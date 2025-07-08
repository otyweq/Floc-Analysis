import os
from PIL import Image, ImageOps
import numpy as np
import cv2

# 设置输入文件夹和输出文件夹
input_folder = 'D:\\PythonProjects\\95_矾花识别\\新实验\\第一组\\原始图像'
output_folder_gray = 'D:\\PythonProjects\\95_矾花识别\\新实验\\第一组\\灰度图像'
output_folder_thresh = 'D:\\PythonProjects\\95_矾花识别\\新实验\\第一组\\阈值分割图像'

# 创建输出文件夹，如果不存在
os.makedirs(output_folder_gray, exist_ok=True)
os.makedirs(output_folder_thresh, exist_ok=True)

# 应用加权平均值法进行灰度处理
def weighted_gray(image):
    gray = ImageOps.grayscale(image)
    return gray

# 批量处理文件夹中的所有BMP图片
for filename in os.listdir(input_folder):
    if filename.endswith('.bmp'):
        image_path = os.path.join(input_folder, filename)

        # 确保文件存在
        if not os.path.exists(image_path):
            continue

        # 读取BMP图像并处理
        try:
            with Image.open(image_path) as img:
                # 将RGB图像转换为灰度图像
                gray_image = weighted_gray(img)
                gray_image_array = np.array(gray_image)

                # 使用Otsu's方法进行阈值分割
                _, thresh_image_array = cv2.threshold(gray_image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh_image = Image.fromarray(thresh_image_array)

                # 保存处理后的图像
                gray_image_path = os.path.join(output_folder_gray, f'gray_{filename.replace(".bmp", ".png")}')
                thresh_image_path = os.path.join(output_folder_thresh, f'thresh_{filename.replace(".bmp", ".png")}')

                gray_image.save(gray_image_path, 'PNG')
                thresh_image.save(thresh_image_path, 'PNG')

        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            continue

print('Batch processing completed.')
