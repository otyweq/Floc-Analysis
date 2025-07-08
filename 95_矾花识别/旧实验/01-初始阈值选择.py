import cv2
from PIL import Image

im = Image.open('采集图像/第1组057/第一组1-15-1.tif')
im.save('converted_image.png')
image = cv2.imread('converted_image.png')
if image is not None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print("Calculated threshold: ", _)
    cv2.imshow("Binary Image", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load the converted image")


# # 读取图像，转换为灰度图
# image = cv2.imread('./采集图像/第1组057/第一组1-15-1.tif')
# if image is None:
#     print("Error loading image")
# else:
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 自动计算阈值并应用
