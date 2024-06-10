
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("EdgeDetection/CannyEdgeDetection/imgs/dog.jpg")

# 彩色图片转换为灰度图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯滤波，减少噪声对边缘检测的影响，
blurred = cv2.GaussianBlur(gray,(5,5),1.4) # 图片，高斯核，高斯核在 x 方向上的标准差，y方向没有指定的话则和x的一致

# 使用Sobel算子计算图像在x和y方向上的梯度
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# 计算x和y方向的梯度幅值和方向来得到边缘强度和方向
magnitude = cv2.magnitude(grad_x, grad_y) # 梯度幅值
angle = cv2.phase(grad_x, grad_y, angleInDegrees=True) # 梯度方向

# 扫描整幅图像，去除非边缘点，保留局部梯度极大值的点
angle = angle % 180
output = np.zeros_like(magnitude, dtype=np.uint8)
for i in range(1, magnitude.shape[0] - 1):
    for j in range(1, magnitude.shape[1] - 1):
        q = 255
        r = 255
        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
            q = magnitude[i, j + 1]
            r = magnitude[i, j - 1]
        elif 22.5 <= angle[i, j] < 67.5:
            q = magnitude[i + 1, j - 1]
            r = magnitude[i - 1, j + 1]
        elif 67.5 <= angle[i, j] < 112.5:
            q = magnitude[i + 1, j]
            r = magnitude[i - 1, j]
        elif 112.5 <= angle[i, j] < 157.5:
            q = magnitude[i - 1, j - 1]
            r = magnitude[i + 1, j + 1]

        if magnitude[i, j] >= q and magnitude[i, j] >= r:
            output[i, j] = magnitude[i, j]
        else:
            output[i, j] = 0

# 对梯度幅值应用双阈值以确定强边缘和潜在边缘
high_threshold = output.max() * 0.09
low_threshold = high_threshold * 0.5
strong_edges = (output >= high_threshold).astype(np.uint8)
weak_edges = ((output >= low_threshold) & (output < high_threshold)).astype(np.uint8)

# 边缘连接（滞后阈值）
# 通过连接强边缘和弱边缘来完成边缘检测
edges = np.zeros_like(output, dtype=np.uint8)
for i in range(1, output.shape[0] - 1):
    for j in range(1, output.shape[1] - 1):
        if strong_edges[i, j]:
            edges[i, j] = 255
        elif weak_edges[i, j]:
            if (strong_edges[i + 1, j - 1:j + 2].any() or
                strong_edges[i - 1, j - 1:j + 2].any() or
                strong_edges[i, [j - 1, j + 1]].any()):
                edges[i, j] = 255

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Canny Edge Detection')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()

if __name__ == "main":
    pass