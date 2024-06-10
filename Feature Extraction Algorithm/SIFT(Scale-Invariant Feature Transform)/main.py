import cv2
import matplotlib.pyplot as plt

# SIFT（Scale-Invariant Feature Transform）是一种经典的特征提取算法，用于检测和描述图像中的局部特征。
# 它具有尺度不变性和旋转不变性，在计算机视觉和图像处理领域应用广泛

def sift_feature_extraction(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 创建SIFT对象
    sift = cv2.SIFT_create()

    # 检测关键点和计算特征描述子
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 在图像上绘制关键点
    output_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return output_image, keypoints, descriptors

# 图像路径
image_path = 'imgs/dog.jpg'

# 进行SIFT特征提取
output_image, keypoints, descriptors = sift_feature_extraction(image_path)

# 显示结果
plt.figure(figsize=(10, 5))
plt.title('SIFT Feature Extraction')
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print(f'Number of keypoints detected: {len(keypoints)}')
print(f'Descriptors shape: {descriptors.shape}')
print(f"descriptors:{descriptors}")
