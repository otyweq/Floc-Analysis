import cv2
import os

def determine_initial_threshold(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 使用Otsu的方法自动确定阈值
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# 替换为您的代表性图像路径
representative_image_path = 'path_to_representative_image.jpg'
initial_threshold = determine_initial_threshold(representative_image_path)
print(f"Determined initial threshold: {initial_threshold}")

def batch_process_images_with_threshold(directory_path, threshold, output_directory=None):
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # 应用确定的初始阈值进行二值化处理
            _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

            if output_directory:
                save_path = os.path.join(output_directory, f"thresholded_{filename}")
                cv2.imwrite(save_path, binary_image)

    print("Batch processing with initial threshold completed.")


# 替换为您的图像目录路径和输出目录路径
input_directory = 'path_to_your_image_directory'
output_directory = 'path_to_your_output_directory'
batch_process_images_with_threshold(input_directory, initial_threshold, output_directory)
