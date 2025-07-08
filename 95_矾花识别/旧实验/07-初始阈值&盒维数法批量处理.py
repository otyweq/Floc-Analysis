import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops


def box_count(img, min_dim=1, max_dim=10, n_dims=10):
    sizes = np.floor(np.geomspace(min_dim, max_dim, n_dims)).astype(int)
    counts = []
    for size in sizes:
        resized_img = cv2.resize(img, (int(img.shape[1] / size), int(img.shape[0] / size)))
        count = np.count_nonzero(resized_img)
        counts.append(count)
    return sizes, counts


def calculate_fractal_dimension(sizes, counts):
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return -coeffs[0]


def process_images_from_directory(directory_path):
    # List all files in the directory
    image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply Otsu's thresholding
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        sizes, counts = box_count(binary_image)
        fractal_dimension = calculate_fractal_dimension(sizes, counts)

        print(f"Image: {image_file}, Fractal Dimension: {fractal_dimension:.4f}")

        # Optional: Save or display the binary image
        # cv2.imwrite(f'binary_{image_file}', binary_image)
        # cv2.imshow('Binary Image', binary_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


# Replace 'path_to_your_directory' with your directory containing images
process_images_from_directory('path_to_your_directory')
