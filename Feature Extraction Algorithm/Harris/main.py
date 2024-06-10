import cv2
import numpy as np
import matplotlib.pyplot as plt

# Harris角点检测算法是一种用于检测图像中角点的特征检测算法。
# 角点是图像中特征丰富且变化较大的点，在图像配准、物体识别等任务中具有重要意义。

def harris_corner_detection(image_path, block_size=2, ksize=3, k=0.04):
    # 读取图像并转换为灰度图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 转换为float32类型
    gray = np.float32(gray)
    
    # 计算Harris角点检测响应矩阵
    harris_response = cv2.cornerHarris(gray, block_size, ksize, k)
    
    # 扩展Harris角点响应图，便于显示
    harris_response = cv2.dilate(harris_response, None)
    
    # 设置阈值并标记角点
    threshold = 0.01 * harris_response.max()
    image[harris_response > threshold] = [0, 0, 255]
    
    return image, harris_response

# 图像路径
image_path = 'imgs/dog.jpg'

# 进行Harris角点检测
output_image, harris_response = harris_corner_detection(image_path)

# 显示结果
plt.figure(figsize=(10, 5))
plt.title('Harris Corner Detection')
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
