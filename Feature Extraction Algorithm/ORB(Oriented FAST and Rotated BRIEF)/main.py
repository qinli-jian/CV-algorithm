import cv2
import matplotlib.pyplot as plt

# ORB（Oriented FAST and Rotated BRIEF）是一种高效的特征检测和描述算法，
# 它结合了FAST关键点检测器和BRIEF描述子，并进行了增强，使其具有旋转不变性。
# ORB算法具有较高的计算效率和较好的特征匹配性能，是SIFT和SURF的有效替代方案。

def orb_feature_extraction(image_path):
    # 读取图像并转换为灰度图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 创建ORB对象
    orb = cv2.ORB_create()

    # 检测关键点
    keypoints = orb.detect(gray, None)

    # 计算特征描述子
    keypoints, descriptors = orb.compute(gray, keypoints)

    # 在图像上绘制关键点
    output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

    return output_image, keypoints, descriptors

# 图像路径
image_path = 'imgs/dog.jpg'

# 进行ORB特征提取
output_image, keypoints, descriptors = orb_feature_extraction(image_path)

# 显示结果
plt.figure(figsize=(10, 5))
plt.title('ORB Feature Extraction')
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print(f'Number of keypoints detected: {len(keypoints)}')
print(f'Descriptors shape: {descriptors.shape}')
