import cv2
import numpy as np
import matplotlib.pyplot as plt

def roberts_edge_detection(image):
    # 灰度化图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 定义Roberts算子
    roberts_kernel_x = np.array([[1, 0], [0, -1]])
    roberts_kernel_y = np.array([[0, 1], [-1, 0]])

    # 应用Roberts算子
    grad_x = cv2.filter2D(gray, cv2.CV_64F, roberts_kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_64F, roberts_kernel_y)

    # 计算梯度幅值
    magnitude = cv2.magnitude(grad_x, grad_y)

    # 将梯度幅值归一化到0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.uint8(magnitude)

    return magnitude

# 读取图像
image = cv2.imread('EdgeDetection/RobertsEdgeDetection/imgs/dog.jpg')

# 进行Roberts边缘检测
edges = roberts_edge_detection(image)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Roberts Edge Detection')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()
