import cv2
import numpy as np


def process_floc_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 灰度化预处理，使用加权平均法（已在cvtColor中完成）

    # 阈值分割
    _, thresh = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)

    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 絮体数量
    floc_count = len(contours)
    print(f"Detected flocs: {floc_count}")

    # 可视化结果
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Flocs", image)
    cv2.imshow("Thresholded", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 替换为您的图像路径
image_path = '/path/to/your/image.jpg'
process_floc_image(image_path)
