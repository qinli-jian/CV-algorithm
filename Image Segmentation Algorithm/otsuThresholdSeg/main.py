import cv2
import numpy as np
import matplotlib.pyplot as plt

def otsu_threshold(image):
    # 1. 计算直方图
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # 2. 计算归一化直方图
    hist_norm = hist / float(hist.sum())

    # 3. 计算累计和
    cumulative_sum = np.cumsum(hist_norm)

    # 4. 计算累计均值
    cumulative_mean = np.cumsum(hist_norm * np.arange(256))

    # 5. 计算类间方差
    global_mean = cumulative_mean[-1]
    between_class_variance = ((global_mean * cumulative_sum - cumulative_mean) ** 2) / (cumulative_sum * (1 - cumulative_sum))
    
    # 6. 找到最大类间方差对应的阈值
    optimal_threshold = np.argmax(between_class_variance)

    # 7. 应用阈值进行分割
    _, binary_image = cv2.threshold(image, optimal_threshold, 255, cv2.THRESH_BINARY)

    return optimal_threshold, binary_image

# 读取图像
image = cv2.imread('imgs/dog.jpg', cv2.IMREAD_GRAYSCALE)

# 进行Otsu阈值分割
optimal_threshold, binary_image = otsu_threshold(image)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Otsu Thresholding')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.show()

print(f'Optimal Threshold: {optimal_threshold}')
