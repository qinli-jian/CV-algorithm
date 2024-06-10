import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_shift_segmentation(image_path, sp, sr):
    # 1. 读取图像并转换颜色空间
    image = cv2.imread(image_path)
    image_luv = cv2.cvtColor(image, cv2.COLOR_BGR2Luv) # 转换为Luv颜色空间
    
    # 2. 准备数据
    flat_image = image_luv.reshape((-1, 3))
    flat_image = np.float32(flat_image)
    
    # 3. 应用Mean Shift算法，sp参数是空间窗口半径，sr参数是色彩窗口半径。这个函数会返回处理后的图像。
    # Mean Shift算法是一种无监督的聚类算法，主要用于图像分割和目标跟踪。它通过在特征空间中寻找高密度区域，将数据点移动到密度最大的位置，逐步收敛，最终实现分割
    segmented_image = cv2.pyrMeanShiftFiltering(image, sp, sr)
    
    return segmented_image

# 图像路径
image_path = 'imgs/dog.jpg'

# 设置Mean Shift参数
sp = 20  # 空间窗口半径
sr = 40  # 色彩窗口半径

# 进行Mean Shift分割
segmented_image = mean_shift_segmentation(image_path, sp, sr)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Mean Shift Segmentation')
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
