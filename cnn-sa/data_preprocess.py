import os

from PIL import Image
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
import pickle
import cv2


def get_np_from_ply(path):
    plydata = PlyData.read(path)  # 读取文件
    data = plydata.elements[0].data  # 读取数据
    data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
    data_np = np.zeros(data_pd.shape, dtype=np.float32)  # 初始化储存数据的array
    property_names = data[0].dtype.names  # 读取property的名字
    for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
        data_np[:, i] = data_pd[name]
    return data_np[:, :3]


def get_rgb_img(path):
    rgb_img = cv2.imread(path)
    return rgb_img


def get_depth_img(path):
    depth_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return depth_img


def get_img(path):
    img = Image.open(path)
    return img


def main():
    noise_path = r"D:\documents\PythonProjects\Wing_process\dataset\wing_09(5000_smooth)\input_pcd(noise)"
    point_path = r"D:\documents\PythonProjects\Wing_process\dataset\wing_09(5000_smooth)\control_points"
    # point_data = get_np_from_ply(r"E:\ChaoShihan\Wing_process\dataset\wing_10(10000-LL)\control_points\control_points.ply")
    color_path = r"D:\documents\PythonProjects\Wing_process\dataset\wing_09(5000_smooth)\color"
    depth_path = r"D:\documents\PythonProjects\Wing_process\dataset\wing_09(5000_smooth)\depth"
    # label_path = r'E:\ChaoShihan\Wing_process\dataset\wing_05\corres_pcd\output_pcd'

    label = get_np_from_ply(r"D:\documents\PythonProjects\Wing_process\dataset\wing_09(5000_smooth)\output_pcd\wing_output.ply")

    all_data = []
    for i in range(5000):
        file_id = f'{i:06d}'
        noise_data = get_np_from_ply(os.path.join(noise_path, f'wing_input(noise)_{file_id}.ply'))
        point_data = get_np_from_ply(os.path.join(point_path, f'control_points_{file_id}.ply'))
        color_data = get_rgb_img(os.path.join(color_path, f'color_{file_id}.png'))
        depth_data = get_depth_img(os.path.join(depth_path, f'depth_{file_id}.png'))
        # label = get_np_from_ply(os.path.join(label_path, f'wing_output_{file_id}.ply'))
        all_data.append({
            "noise_data": noise_data,
            "point_data": point_data,
            "color_data": color_data,
            "depth_data": depth_data,
            "label": label
        })

    pickle.dump(all_data, open('dataset/all_data_wing_noise(SO-smooth)_5000.pkl', 'wb'))


if __name__ == '__main__':
    main()
