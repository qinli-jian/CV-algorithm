import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('EdgeDetection/PrewittEdgeDetection/imgs/dog.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 手动定义Prewitt算子，
prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

# 应用Prewitt算子
grad_x = cv2.filter2D(gray, cv2.CV_64F, prewitt_kernel_x)
grad_y = cv2.filter2D(gray, cv2.CV_64F, prewitt_kernel_y)

# 计算梯度幅值和方向
magnitude = cv2.magnitude(grad_x, grad_y)

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
plt.title('Prewitt Edge Detection')
plt.imshow(magnitude, cmap='gray')
plt.axis('off')

plt.show()
