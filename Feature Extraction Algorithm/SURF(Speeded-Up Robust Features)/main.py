import cv2
import matplotlib.pyplot as plt

def surf_feature_extraction(image_path):
    # 读取图像并转换为灰度图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 创建SURF对象（确保OpenCV版本支持）
    surf = cv2.xfeatures2d.SURF_create(400)  # Hessian阈值为400

    # 检测关键点和计算特征描述子
    keypoints, descriptors = surf.detectAndCompute(gray, None)

    # 在图像上绘制关键点
    output_image = cv2.drawKeypoints(image, keypoints, None, (255, 0, 0), 4)

    return output_image, keypoints, descriptors

# 图像路径
image_path = 'imgs/dog.jpg'

# 进行SURF特征提取
output_image, keypoints, descriptors = surf_feature_extraction(image_path)

# 显示结果
plt.figure(figsize=(10, 5))
plt.title('SURF Feature Extraction')
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print(f'Number of keypoints detected: {len(keypoints)}')
print(f'Descriptors shape: {descriptors.shape}')
