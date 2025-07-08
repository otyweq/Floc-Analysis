import cv2
import numpy as np
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt


def box_count(img, min_dim=1, max_dim=10, n_dims=10):
    """
    Perform the box-counting method on a binary image.
    :param img: Binary image
    :param min_dim: Minimum dimension of the box
    :param max_dim: Maximum dimension of the box
    :param n_dims: Number of dimensions to calculate
    :return: Dimensions and counts
    """
    sizes = np.floor(np.geomspace(min_dim, max_dim, n_dims)).astype(int)
    counts = []

    for size in sizes:
        # Resizing the image
        resized_img = cv2.resize(img, (int(img.shape[1] / size), int(img.shape[0] / size)))
        # Counting the non-zero (i.e., white) pixels
        count = np.count_nonzero(resized_img)
        counts.append(count)

    return sizes, counts


def calculate_fractal_dimension(sizes, counts):
    """
    Calculate the fractal dimension given sizes and counts.
    :param sizes: Sizes of the boxes
    :param counts: Counts of the non-zero pixels
    :return: Fractal dimension
    """
    # Taking the logarithm of both sizes and counts
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    # Linear regression
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return -coeffs[0]


def main(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)

    # Perform box counting
    sizes, counts = box_count(binary_image)
    # Calculate fractal dimension
    fractal_dimension = calculate_fractal_dimension(sizes, counts)

    print(f"Fractal Dimension: {fractal_dimension:.4f}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.loglog(sizes, counts, 'o-', basex=10, basey=10)
    plt.title('Log-Log Plot of Box Count')
    plt.xlabel('Size of Box')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()


# Replace 'path_to_your_image.jpg' with your image file path
main('path_to_your_image.jpg')
