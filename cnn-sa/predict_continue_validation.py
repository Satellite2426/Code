import torch
import torch.nn as nn
import torch.nn.functional as F
import pyvista as pv
from PIL import Image
import numpy as np
from model_point_three_layers import Net
from torchvision import transforms as transforms
from datasets import MyDataset
from torch.utils.data import Dataset
from plyfile import PlyData, PlyElement
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KDTree
from pyvista import themes

pv.set_plot_theme(themes.DocumentTheme())

device = ("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model = Net()
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load("model_pt/model_noise5000(SO-CNN-SA_smooth(20))_2023-07-03_20-02-39.pt"))

    # model.eval()  # 关闭 model BN层的训练模式

    all_data = pickle.load(
        open(
            r"D:\documents\PythonProjects\ChaoShihan20\cnn-sa\dataset\wing_noise(SO-smooth(20)_control_points)_1000.pkl",
            'rb'))
    # data_index = int(len(all_data) * 0.8)
    # test_data = all_data[data_index:]
    test_data = all_data[:]

    test_loader = torch.utils.data.DataLoader(MyDataset(test_data), batch_size=1)

    index = 0
    distances = []
    meshed = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            noise_data, point_data, color_data, depth_data, label = [i.to(device) for i in data]
            output = model(noise_data, point_data, color_data, depth_data)

            result = output.squeeze().cpu().numpy() * 100
            predict_pcd = pv.PolyData(result)

            label_pcd = pv.read(
                r"D:\documents\PythonProjects\Wing_process\dataset\wing_12(5000)\output_pcd\wing_output.ply")
            predict_points = predict_pcd.points
            label_points = label_pcd.points
            tree = KDTree(label_points)
            distance, indices = tree.query(predict_points)
            diff = np.linalg.norm(predict_points - label_points, axis=1)

            # distances.extend(diff.tolist())


            # 将设想将'scalars'限制在0到0.5之间
            # scalars = label_pcd.point_arrays['diff']
            # scalars = np.clip(scalars, 0, 0.5)

            # 创建颜色映射对象并设置范围
            cmap = 'jet'
            lut = pv.LookupTable(cmap)
            lut.scalar_range = (0, 0.6)


            label_pcd.point_arrays.append(diff, 'diff')
            p = pv.Plotter()
            p.add_mesh(label_pcd, scalars='diff', cmap=lut)
            p.show()

            # cloud = pv.PolyData(predict_points)
            # cloud.point_arrays.append(diff, 'diff')
            # p = pv.Plotter()
            # p.add_mesh(cloud, scalars='diff', cmap='jet')
            # p.show()

    # df = pd.DataFrame({'distances': distances})
    # df.to_csv('distances(CNN-SA-3-layers-trans).csv', index=False)
