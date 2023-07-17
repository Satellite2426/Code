from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pickle
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datasets import MyDataset
from model_CA_less import Net
import datetime
import time
import pandas as pd

plt.rcParams['backend'] = 'Agg'


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = []
    train_rmse = []
    for batch_idx, data in enumerate(train_loader):
        noise_data, point_data, color_data, depth_data, label = [i.to(device) for i in data]

        optimizer.zero_grad()
        output = model(noise_data, point_data, color_data, depth_data)
        loss = F.l1_loss(output, label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        train_rmse.append(
            np.sqrt(mean_squared_error(label.cpu().numpy().flatten() * 100,
                                       output.detach().cpu().numpy().flatten() * 100)))
    return np.mean(train_loss), np.mean(train_rmse)


def test(model, device, test_loader):
    model.eval()
    test_loss = []
    test_rmse = []
    # correct = 0
    with torch.no_grad():
        for data in test_loader:
            noise_data, point_data, color_data, depth_data, label = [i.to(device) for i in data]
            output = model(noise_data, point_data, color_data, depth_data)
            test_loss.append(F.mse_loss(output, label, reduction='sum').item())  # sum up batch loss
            # test_loss.append(F.huber_loss(output, label, reduction='sum').item())
            test_rmse.append(
                np.sqrt(mean_squared_error(label.cpu().numpy().flatten() * 100,
                                           output.detach().cpu().numpy().flatten() * 100)))

    test_loss = np.mean(test_loss)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss, np.mean(test_rmse)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=9, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        # test_kwargs.update(cuda_kwargs)

    all_data = pickle.load(
        open(r"C:\Users\Administrator\Desktop\ChaoShihan20\cnn-sa\dataset\all_data_wing_noise(SO-smooth(20))_5000.pkl",
             'rb'))  # 加载自己生成的训练集
    train_idx = int(len(all_data) * 0.6)  # 6:2:2(train:test:verify)
    train_data = all_data[:train_idx]
    test_data = all_data[train_idx: train_idx + int(len(all_data) * 0.2)]

    train_loader = torch.utils.data.DataLoader(MyDataset(train_data), **train_kwargs)  # 创建torch数据集
    test_loader = torch.utils.data.DataLoader(MyDataset(test_data), **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 优化器

    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)  # 每10个epoch学习率乘以0.7
    train_loss_list, train_rmse_list, test_loss_list, test_rmse_list = [], [], [], []  # 记录loss 和 rmse
    for epoch in range(1, args.epochs + 1):
        train_loss, train_rmse = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_rmse = test(model, device, test_loader)

        train_loss_list.append(train_loss)
        train_rmse_list.append(train_rmse)
        test_loss_list.append(test_loss)
        test_rmse_list.append(test_rmse)
        print(f"epoch: {epoch}\n"
              f"train loss: {train_loss}\ttrain rmse: {train_rmse}\n"
              f"test loss: {test_loss}\ttest rmse: {test_rmse}")
        scheduler.step()

    # add time stamp
    now = datetime.datetime.now()
    time_stamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    plt.plot(range(len(train_loss_list)), train_loss_list, 'g-', label=u'train loss')
    plt.savefig(f'result_img/train/trainloss_{time_stamp}.jpg')
    plt.cla()  # 清除当前轴

    train_loss_df = pd.DataFrame({'train loss': train_loss_list})
    train_loss_df.to_csv(f'result_csv/train/train_loss_{time_stamp}.csv', index=False)

    plt.plot(range(len(test_loss_list)), test_loss_list, 'r-', label=u'test loss')
    plt.savefig(f'result_img/test/testloss_{time_stamp}.jpg')
    plt.cla()

    test_loss_df = pd.DataFrame({'test loss': test_loss_list})
    test_loss_df.to_csv(f'result_csv/test/test_loss_{time_stamp}.csv', index=False)

    plt.plot(range(len(train_rmse_list)), train_rmse_list, 'g-', label=u'train rmse')
    plt.savefig(f'result_img/train/trainrmse_{time_stamp}.jpg')
    plt.cla()

    train_rmse_df = pd.DataFrame({'train rmse': train_rmse_list})
    train_rmse_df.to_csv(f'result_csv/train/train_rmse_{time_stamp}.csv', index=False)

    plt.plot(range(len(test_rmse_list)), test_rmse_list, 'r-', label=u'test rmse')
    plt.savefig(f'result_img/test/testrmse_{time_stamp}.jpg')
    plt.cla()

    test_rmse_df = pd.DataFrame({'test rmse': test_rmse_list})
    test_rmse_df.to_csv(f'result_csv/test/test_rmse_{time_stamp}.csv', index=False)

    if args.save_model:
        torch.save(model.state_dict(), f"model_pt/model_noise5000(SO-CNN-CA(output-less)_smooth(20))_{time_stamp}.pt")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = (end_time - start_time) / 60.0
    print(f"Training took :{total_time:.2f} minutes")
