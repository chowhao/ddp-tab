# coding: utf-8
import os
import time
import seaborn as sns
import argparse
import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
from matplotlib import pyplot as plt


# 自定义的数据预处理与模型包
from pkg.model import model
from pkg.loader import data_loader
from pkg.loader import tensor_loader

# ip and port
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '2000'
os.environ['WORLD_SIZE'] = '4'

# parameters
wgt_dcy = 0.1
learn_rate = 0.0001
cost = nn.MSELoss(reduction='mean')
epoch_num = 160
# 实际速度放大100倍, loss除以1e4
multiple = 100
# 数据采样率
f = 0.2


def train(models, dataset, device, epoch):
    models.train()

    loss_all = 0.0
    optimizer = th.optim.Adam(models.parameters(), lr=learn_rate,
                              weight_decay=wgt_dcy)

    for batch, data in enumerate(dataset):
        # 得到速度和标签
        features, label = data
        feature_ctg, feature_ctn = split(features)
        label = label.to(th.float32)
        # 将数据发到设备
        feature_ctg = feature_ctg.to(device)
        feature_ctn = feature_ctn.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = models(feature_ctg, feature_ctn)
        loss = cost(output, label)
        # 累加损失
        loss_all += loss

        loss.backward()
        optimizer.step()

    loss_all /= len(dataset)*multiple*multiple
    print("epoch: {}/{}, train loss = {:.6f}".format(epoch + 1, epoch_num, loss_all))


def valid(models, dataset):
    models.eval()
    total_loss = 0.0

    with th.no_grad():
        for i, data in enumerate(dataset, 0):
            # 得到数据和标签
            features, label = data
            feature_ctg, feature_ctn = split(features)
            label = label.to(th.float32)

            # 预测
            pre = models(feature_ctg, feature_ctn)
            total_loss += cost(pre, label).item()
            # print(i)

    total_loss /= len(dataset)*multiple*multiple

    print("valid loss = {:.6f}".format(total_loss))


# 画出曲线图
def test(models, data_features, data_label):
    models.eval()

    # bug但结果正确, 输入特征里的追踪速度
    test_spd = data_features[:, 2] - 0

    feature_ctg, feature_ctn = split(data_features)
    pred_label = models(feature_ctg, feature_ctn)
    pred_label = pred_label.data.numpy()

    # 加速度累加得到速度, 采样间隔0.2s
    pred_spd = np.zeros(len(test_spd))
    for i in range(len(test_spd)):
        pred_spd[i] = f * sum(pred_label[0:i])

    plot(data_label/multiple, pred_label/multiple, "acceleration")
    plot(test_spd/multiple, pred_spd/multiple, "speed")

    return np.array(test_spd), np.array(pred_spd)


# 画出对比图
def plot(test_data, pred_data, title):
    plt.figure(figsize=(16, 8))
    plt.title('Train ' + title, fontsize=20)
    plt.xlabel('Time / s', fontsize=20)
    if title == "acceleration":
        plt.ylabel(title + r' $ m/s^2 $', fontsize=20)
    else:
        plt.ylabel(title + ' m/s', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.gca()
    # plt.gca().set_box_aspect((4, 1))

    data_len = len(test_data) - 1
    x = f * np.linspace(0, data_len, data_len + 1, endpoint=True)

    plt.plot(x, test_data, 'b', linewidth=1, label="Real "+title)
    plt.plot(x, pred_data, 'r', linewidth=1, label="Predictive "+title)

    plt.legend(loc="upper right", fontsize=20)
    plt.savefig(title)
    # plt.show()


# 速度损失分布图
def pltLos(tst, pred):
    new = tst
    for i in range(len(tst)):
        # new[i] = tst[i]-pred[i]
        new[i] = abs(tst[i]-pred[i])

    plt.figure(figsize=(16, 8))
    sns.histplot(new, kde=True, stat="density",
                 kde_kws=dict(cut=3), alpha=.4,
                 edgecolor=(1, 1, 1, .4)
                 )
    # plt.figure(figsize=(16, 8))
    plt.title('Loss Distribution')
    plt.xlabel('Speed Error m / s')
    plt.ylabel('Percent')
    plt.savefig('loss')


# 将输入特征分成离散和连续
def split(features):
    feature_spt = th.split(features, 3, dim=1)
    feature_ctg = feature_spt[0]
    feature_ctn = feature_spt[1].to(th.float32)
    return feature_ctg, feature_ctn


def main():
    parser = argparse.ArgumentParser(description='PyTorch Distributed Transformer Example')
    parser.add_argument('--backend', type=str, help='Distributed backend',
                        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                        default=dist.Backend.GLOO)
    # parser.add_argument('--init-method', default=None, type=str,
    #                     help='Distributed init_method')
    parser.add_argument('--rank', default=-1, type=int,
                        help='Distributed rank')
    # parser.add_argument('--world-size', default=-1, type=int,
    #                     help='Distributed world_size')
    args = parser.parse_args()

    dist.init_process_group(backend=args.backend,
                            # init_method=args.init_method,
                            rank=args.rank,
                            # world_size=args.world_size
                            )

    my_device = th.device("cuda" if th.cuda.is_available() else "cpu")
    flag = dist.is_initialized()
    print("train device is {}, rank is {}".format(my_device, args.rank))
    print("distributed data parallel is", flag)
    model_dis = model.to(my_device)
    model_dis = nn.parallel.DistributedDataParallel(model_dis)

    path = '../data/cbtc/csv/new/1.hcz2cq/'
    train_path = path + 'train.csv'
    test_path = path + 'test.csv'

    # csv2tensor
    train_features, train_label = tensor_loader(train_path)
    valid_features, valid_label = tensor_loader(test_path)
    # print(valid_features[:, 2])
    # 数据集
    train_dataset = data_loader(train_features, train_label, flag)
    valid_dataset = data_loader(valid_features, valid_label, False)

    # 训练
    start_time = time.time()
    for epoch in range(epoch_num):
        train(model_dis, train_dataset, my_device, epoch)

    print(time.time()-start_time)

    # 测试与验证
    valid(model_dis, valid_dataset)
    # test_data, pred_data = test(model_dis, valid_features, valid_label)
    # pltLos(test_data/multiple, pred_data/multiple)


if __name__ == '__main__':
    main()
