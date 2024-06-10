import cv2
import numpy as np
import matplotlib.pyplot as plt

# ECC（Enhanced Correlation Coefficient）算法是一种用于图像配准（图像对齐）的算法，通过最大化图像间的相关系数来实现最佳配准。
# OpenCV提供了对ECC算法的实现，允许我们在图像间进行亚像素级的配准。
def ecc_image_registration(image_path_1, image_path_2):
    # 读取图像并转换为灰度图像
    image1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)

    # 获取图像的尺寸
    size = image1.shape

    # 初始化仿射变换矩阵
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # 定义ECC算法的终止条件
    number_of_iterations = 5000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # 使用ECC算法计算变换矩阵
    cc, warp_matrix = cv2.findTransformECC(image1, image2, warp_matrix, cv2.MOTION_AFFINE, criteria)

    # 应用变换矩阵对图像进行对齐
    aligned_image = cv2.warpAffine(image2, warp_matrix, (size[1], size[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return image1, image2, aligned_image, warp_matrix

# 图像路径
image_path_1 = 'imgs/dog.jpg'
image_path_2 = 'imgs/dog.jpg'

# 进行ECC图像配准
image1, image2, aligned_image, warp_matrix = ecc_image_registration(image_path_1, image_path_2)

# 显示结果
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Reference Image')
plt.imshow(image1, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Image to Align')
plt.imshow(image2, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Aligned Image')
plt.imshow(aligned_image, cmap='gray')
plt.axis('off')

plt.show()

print("Warp Matrix:\n", warp_matrix)
