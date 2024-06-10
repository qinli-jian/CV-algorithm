import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_contours(image):
    # 1. 读取图像并灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 4. 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. 绘制轮廓
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    return contour_image, contours

# 读取图像
image = cv2.imread('imgs/dog.jpg')

# 进行轮廓检测
contour_image, contours = detect_contours(image)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Contour Detection')
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

print(f'Number of contours detected: {len(contours)}')
