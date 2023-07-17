import numpy as np
import pyvista as pv
from pyvista import themes
import pandas as pd
import os
from sklearn.neighbors import KDTree

pv.set_plot_theme(themes.DocumentTheme())

if __name__ == "__main__":
    label_pcd = pv.read(r"D:\documents\PythonProjects\Wing_process\dataset\wing_12(5000)\output_pcd\wing_output.ply")
    label_points = label_pcd.points
    input_pcd_dir = r"D:\documents\PythonProjects\Wing_process\dataset\wing_12(5000)\input_pcd(noise)"

    distances = []
    for i in range(1000):
        input_pcd = pv.read(os.path.join(input_pcd_dir, f"wing_input(noise)_{i + 4000:06}.ply"))
        input_pcd_points = input_pcd.points
        tree = KDTree(label_points)
        distance, indices = tree.query(input_pcd_points)
        diff = np.linalg.norm(input_pcd_points - label_points, axis=1)

        distances.extend(diff.tolist())

        # label_pcd.point_arrays.append(diff, 'diff')
        # p = pv.Plotter()
        # p.add_mesh(label_pcd, scalars='diff', cmap='jet')
        # p.show()

    df = pd.DataFrame({'distances': distances})
    df.to_csv('distances(input_output_diff).csv', index=False)
