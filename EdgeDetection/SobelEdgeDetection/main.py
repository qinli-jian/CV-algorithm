import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('EdgeDetection/SobelEdgeDetection/imgs/dog.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算Sobel梯度
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # x方向的梯度
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # y方向的梯度

# 计算梯度幅值和方向
magnitude = cv2.magnitude(grad_x, grad_y)  # 梯度幅值
angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)  # 梯度方向

# 将梯度幅值归一化到0-255
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
magnitude = np.uint8(magnitude)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Sobel Edge Detection')
plt.imshow(magnitude, cmap='gray')
plt.axis('off')

plt.show()
