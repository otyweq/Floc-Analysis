import os
from PIL import Image, ImageOps
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd


def process_images(input_folder, output_folder_gray, output_folder_thresh, output_folder_results, output_folder_plots):
    # 创建输出文件夹，如果不存在
    os.makedirs(output_folder_gray, exist_ok=True)
    os.makedirs(output_folder_thresh, exist_ok=True)
    os.makedirs(output_folder_results, exist_ok=True)
    os.makedirs(output_folder_plots, exist_ok=True)

    # 设置全局字体为黑体，适合中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 应用加权平均值法进行灰度处理
    def weighted_gray(image):
        gray = ImageOps.grayscale(image)
        return gray

    # 提取絮体粒径的函数
    def extract_floc_diameter(thresh_image_array):
        contours, _ = cv2.findContours(thresh_image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        diameters = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 0:
                diameter = 2 * np.sqrt(area / np.pi)
                diameters.append(diameter)
        return diameters

    # 计算絮体图像密度
    def calculate_floc_density(thresh_image_array):
        floc_pixels = np.sum(thresh_image_array == 255)
        total_pixels = thresh_image_array.size
        density = floc_pixels / total_pixels
        return density

    # 计算分形维数
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

        # 线性回归拟合
        coeffs = np.polyfit(log_perimeters, log_areas, 1)
        fractal_dimension = coeffs[0]

        # 绘制拟合曲线并保存为图像
        plt.figure()
        plt.plot(log_perimeters, log_areas, 'o', label='数据点')
        plt.plot(log_perimeters, np.polyval(coeffs, log_perimeters), label='拟合线')
        plt.xlabel('周长的对数值')
        plt.ylabel('面积的对数值')
        plt.legend()
        plt.savefig(plot_path)
        plt.close()

        return fractal_dimension

    # 提取絮体数量的函数
    def extract_floc_count(thresh_image_array):
        contours, _ = cv2.findContours(thresh_image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)

    # 裁剪图像中间部分的函数
    def crop_center(image, crop_size):
        """
        裁剪图像的中间部分
        :param image: 输入的PIL图像
        :param crop_size: 裁剪的大小 (宽, 高)
        :return: 裁剪后的图像
        """
        width, height = image.size
        new_width, new_height = crop_size

        # 计算中心区域的左上角坐标和右下角坐标
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = (width + new_width) // 2
        bottom = (height + new_height) // 2

        return image.crop((left, top, right, bottom))

    # 初始化结果列表
    results = []

    # 批量处理文件夹中的所有BMP和TIF图片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.bmp', '.tif')):  # 添加对 .tif 的支持
            image_path = os.path.join(input_folder, filename)

            # 确保文件存在
            if not os.path.exists(image_path):
                continue

            # 读取图像并处理
            try:
                with Image.open(image_path) as img:
                    # 裁剪中间部分
                    cropped_img = crop_center(img, (1000, 1000))  # 裁剪一个1000x1000的中间区域

                    # 将RGB图像转换为灰度图像
                    gray_image = weighted_gray(cropped_img)
                    gray_image_array = np.array(gray_image)

                    # 使用Otsu's方法进行阈值分割
                    _, thresh_image_array = cv2.threshold(gray_image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    thresh_image = Image.fromarray(thresh_image_array)

                    # 提取絮体数量
                    floc_count = extract_floc_count(thresh_image_array)

                    # 提取絮体粒径
                    diameters = extract_floc_diameter(thresh_image_array)
                    avg_diameter = np.mean(diameters) if diameters else 0

                    # 计算絮体图像密度
                    floc_density = calculate_floc_density(thresh_image_array)

                    # 计算分形维数
                    plot_path = os.path.join(output_folder_plots,
                                             f'fractal_{filename.replace(".bmp", ".png").replace(".tif", ".png")}')
                    fractal_dimension = calculate_fractal_dimension(thresh_image_array, plot_path)

                    print(
                        f'文件: {filename}, 絮体数量: {floc_count}, 平均粒径: {avg_diameter}, 密度: {floc_density}, 分形维数: {fractal_dimension}')

                    # 保存处理后的图像
                    gray_image_path = os.path.join(output_folder_gray,
                                                   f'gray_{filename.replace(".bmp", ".png").replace(".tif", ".png")}')
                    thresh_image_path = os.path.join(output_folder_thresh,
                                                     f'thresh_{filename.replace(".bmp", ".png").replace(".tif", ".png")}')

                    gray_image.save(gray_image_path, 'PNG')
                    thresh_image.save(thresh_image_path, 'PNG')

                    # 保存分析结果到列表
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

    # 将结果保存到DataFrame
    df_results = pd.DataFrame(results)
    result_csv_path = os.path.join(output_folder_results, '分析结果.csv')
    df_results.to_csv(result_csv_path, index=False, encoding='utf-8-sig')

    print('批量处理完成，结果已保存到表格中。')


# 主函数，用于调用上述图像处理函数
def main(input_folder):
    # 定义输出文件夹路径
    output_folder_gray = os.path.join(input_folder, '灰度图像')
    output_folder_thresh = os.path.join(input_folder, '阈值分割图像')
    output_folder_results = os.path.join(input_folder, '分析结果')
    output_folder_plots = os.path.join(input_folder, '分形维数图像')

    # 调用处理函数
    process_images(input_folder, output_folder_gray, output_folder_thresh, output_folder_results, output_folder_plots)


# 调用主函数并传入输入文件夹路径
if __name__ == '__main__':
    input_folder = r"D:\大学生活\大创\基于卷积神经网络（CNN）的净水厂混凝除浊图像识别与调控策略研究\中期\实验6"
    main(input_folder)

