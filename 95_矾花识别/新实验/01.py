import os
from PIL import Image, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 设置图像路径
image_path = r'D:\PythonProjects\95_矾花识别\新实验\第一组\原始图像\4.bmp'

# 读取BMP图像并处理
try:
    with Image.open(image_path) as img:

        # 将RGB图像转换为灰度图像
        gray_image = ImageOps.grayscale(img)
        gray_image_array = np.array(gray_image)

        # 使用固定阈值方法进行阈值分割
        _, thresh_fixed = cv2.threshold(gray_image_array, 170, 255, cv2.THRESH_BINARY)

        # 使用Otsu's方法进行阈值分割
        _, thresh_otsu = cv2.threshold(gray_image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 使用自适应阈值方法进行阈值分割
        thresh_adaptive = cv2.adaptiveThreshold(gray_image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)

        # 显示结果
        images = [gray_image_array, thresh_fixed, thresh_otsu, thresh_adaptive]
        titles = ['Gray Image', 'Fixed Threshold', "Otsu's Threshold", 'Adaptive Threshold']

        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()

except Exception as e:
    print(f'Failed to read or process {image_path}. Error: {e}')
