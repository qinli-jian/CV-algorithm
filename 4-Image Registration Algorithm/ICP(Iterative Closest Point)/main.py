# pip install open3d
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# ICP（Iterative Closest Point）算法是一种用于点云对齐的常用方法，广泛应用于3D图像处理和计算机视觉中。
# ICP算法通过迭代地找到两组点云之间的最优刚性变换，使得一个点云尽可能地与另一个点云对齐。

def draw_registration_result(source, target, transformation):
    source_temp = source.copy()
    target_temp = target.copy()
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def icp_registration(source_path, target_path):
    # 读取点云
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    # 初始变换矩阵
    initial_transformation = np.eye(4)

    # 应用初始变换
    source.transform(initial_transformation)

    # ICP算法
    threshold = 0.02  # 距离阈值
    max_iterations = 50  # 最大迭代次数
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    # 打印结果
    print("ICP converged:", result_icp.convergence_criteria)
    print("Fitness:", result_icp.fitness)
    print("Inlier RMSE:", result_icp.inlier_rmse)
    print("Transformation Matrix:\n", result_icp.transformation)

    # 可视化配准结果
    draw_registration_result(source, target, result_icp.transformation)

    return result_icp

# 点云文件路径
source_path = 'source.pcd'
target_path = 'target.pcd'

# 进行ICP配准
result_icp = icp_registration(source_path, target_path)
