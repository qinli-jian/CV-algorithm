import cv2
import numpy as np
import matplotlib.pyplot as plt

# GrabCut算法是一种用于图像前景分割的交互式图像分割算法。它通过结合高斯混合模型（GMM）和图割（Graph Cut）算法，有效地将前景与背景分离。

def grabcut_segmentation(image_path):
    # 1. 读取图像
    image = cv2.imread(image_path)
    
    # 2. 初始化矩形
    height, width = image.shape[:2]
    rect = (int(width*0.1), int(height*0.1), int(width*0.8), int(height*0.8))  # 定义矩形框

    # 3. 初始化掩码
    mask = np.zeros(image.shape[:2], np.uint8)

    # 4. 初始化模型
    bgd_model = np.zeros((1, 65), np.float64)  # 背景模型
    fgd_model = np.zeros((1, 65), np.float64)  # 前景模型

    # 5. 应用GrabCut算法
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 6. 根据处理结果生成输出图像
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask2[:, :, np.newaxis]

    return segmented_image, mask

# 图像路径
image_path = 'imgs/dog.jpg'

# 进行GrabCut分割
segmented_image, mask = grabcut_segmentation(image_path)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('GrabCut Segmentation')
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
