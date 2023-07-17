import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        # CNN for RGB image
        self.cnn_rgb = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),  # inplace 可以选择就地操作
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256)
        )

        # CNN for depth image
        self.cnn_depth = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),  # inplace 可以选择就地操作
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256)
        )

        # LSTM for point cloud
        self.lstm_pointcloud = nn.LSTM(3, 256, num_layers=2, batch_first=True)

        # MLP for control points
        self.mlp_controlpoints = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
        )

        # FC for down sampling
        self.fusion_mlp = nn.Linear(1058, 1036)

        # Output MLP for point cloud
        self.output_mlp = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1),
        )

    def forward(self, noise_data, point_data, color_data, depth_data):
        # CNN for RGB image
        out_rgb = self.cnn_rgb(color_data).unsqueeze(2)  # 1, 256, 40, 30
        # out_rgb = out_rgb.view(out_rgb.size(0), -1)

        # CNN for depth image
        out_depth = self.cnn_depth(depth_data).unsqueeze(2)  # 1, 256, 40, 30
        # out_depth = out_depth.view(out_depth.size(0), -1)

        # LSTM for point cloud
        out_pointcloud, _ = self.lstm_pointcloud(noise_data)  # 1, 1036, 256
        # out_pointcloud = out_pointcloud[:, -1, :]       # 1, 256
        out_pointcloud = out_pointcloud.permute(0, 2, 1)  # 1, 256, 1036

        # MLP for control points
        # out_controlpoints = self.mlp_controlpoints(x_controlpoints.view(x_controlpoints).size(0), -1)
        out_controlpoints = self.mlp_controlpoints(point_data)  # 1, 10, 256
        out_controlpoints = out_controlpoints.permute(0, 2, 1)  # 1, 256, 10

        # Concatenate all feature
        out = torch.cat((out_rgb, out_depth, out_pointcloud, out_controlpoints), dim=2)

        # Fusion MLP
        out = self.fusion_mlp(out)
        out = self.output_mlp(out).permute(0, 2, 1)

        return out


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
#     model = CNN_LSTM().to(device)
#     x_rgb = torch.rand(1, 3, 224, 224).cuda()
#     x_depth = torch.rand(1, 1, 224, 224).cuda()
#     x_pointcloud = torch.rand(1, 1036, 3).cuda()
#     x_controlpoints = torch.rand(1, 20, 3).cuda()
#     output = model(x_pointcloud, x_controlpoints, x_rgb, x_depth)
#     print(output.size())  # 1, 1036, 3
