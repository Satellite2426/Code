from torch import nn
import torch
import torch.nn.functional as F
from attention import ChannelAttention


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.noise_model = nn.Sequential(  # 噪声数据进行特征提取，下采样
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        self.point_model = nn.Sequential(  # 关键点数据的特征提取
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        self.color_model = nn.Sequential(  # color图像的特征提取
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 第一层卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 第一层池化
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 第二层卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 第二层池化
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 第三层卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 第三层池化
            nn.Flatten(),  # 将卷积层输出展平成一维张量
            nn.Linear(224 * 224, 256),  # 全连接层
        )

        self.depth_model = nn.Sequential(  # 深度图片的特征提取
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 第一层卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 第一层池化
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 第二层卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 第二层池化
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 第三层卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 第三层池化
            nn.Flatten(),  # 将卷积层输出展平成一维张量
            nn.Linear(224 * 224, 256),  # 全连接层
        )

        # self.att = nn.MultiheadAttention(256, 2, dropout=0.5)
        # self.attention = Attention(256)  # 自监督注意力机制
        self.attention = ChannelAttention(256)  # 自监督注意力机制
        # self.attention = ResidualAttention()  # 自监督注意力机制
        self.layer1 = nn.Linear(1058, 1036)  # 因为需要输出1036个点，故这地方通过全连接进行采样
        # MLP layers for the whole point cloud
        self.output = nn.Sequential(  # 输出的网络层，最终输出的就是1036， 3
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1),
        )

    def forward(self, noise_data, point_data, color_data, depth_data):
        noise_feature = self.noise_model(noise_data.permute(0, 2, 1))  # b, 256, 1036
        point_feature = self.point_model(point_data.permute(0, 2, 1))  # b, 256, 10
        color_feature = self.color_model(color_data)  # b, 256
        depth_feature = self.depth_model(depth_data)  # b, 256

        concat_feature = torch.cat(  # 特征融合 b,256,1048
            [noise_feature, point_feature, color_feature.unsqueeze(2), depth_feature.unsqueeze(2)], dim=2)

        concat_att = self.attention(concat_feature).permute(0, 2, 1)  # b, 256, 1
        # x = concat_att.size()
        concat_feature = concat_feature * concat_att

        # feature = self.att(noise_feature.permute(0, 2, 1), color_feature.unsqueeze(1), depth_feature.unsqueeze(1))
        concat_feature = self.layer1(concat_feature)
        output = self.output(concat_feature).permute(0, 2, 1)
        return output


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
#     model = Net().to(device)
#     x_rgb = torch.rand(1, 3, 224, 224).cuda()
#     x_depth = torch.rand(1, 1, 224, 224).cuda()
#     x_pointcloud = torch.rand(1, 1036, 3).cuda()
#     x_controlpoints = torch.rand(1, 20, 3).cuda()
#     output = model(x_pointcloud, x_controlpoints, x_rgb, x_depth)
#     print(output.size())  # 1, 1036, 3
