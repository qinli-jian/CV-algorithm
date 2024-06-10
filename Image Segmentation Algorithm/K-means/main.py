import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans_segmentation(image, K):
    # 1. 将图像数据转换为一维向量
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # 2. 定义K-means算法的终止条件和初始质心选择方式
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 3. 将中心值转换为8位整数
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]

    # 4. 重构分割后的图像
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

# 读取图像
image = cv2.imread('imgs/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 设置聚类簇数
K = 4

# 进行K-means聚类分割
segmented_image = kmeans_segmentation(image, K)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('K-means Segmentation')
plt.imshow(segmented_image)
plt.axis('off')

plt.show()
