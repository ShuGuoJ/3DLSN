"""训练深度可分离孪生网络, 并每次保存模型的参数"""
import torch
from torch import nn, optim
import numpy as np
from scipy.io import loadmat
from utils import weight_init
from HSIDataset import HSIDatasetPair
from Model.module import Net_Depthwise
from torch.utils.data import DataLoader
from Trainer import PairTrainer
import os
import argparse
from visdom import Visdom
from sklearn.preprocessing import scale
from Monitor import GradMonitor
from torch.utils.data import random_split
from configparser import ConfigParser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train contrastive model')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='Dataset name')
    parser.add_argument('--epoch', type=int, default=1,
                        help='Training epoch')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpu id')
    parser.add_argument('--patch', type=int, default=7,
                        help='Patch size')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden size')
    parser.add_argument('--batchsz', type=int, default=10)
    parser.add_argument('--run', type=int, default=10)
    parser.add_argument('--spc', type=int, default=10)
    parser.add_argument('--seed', type=int, default=666)
    arg = parser.parse_args()
    config = ConfigParser()
    config.read('dataInfo.ini')
    viz = Visdom(port=17000)
    # 保存根路径
    save_root_path = 'depthwise_models/{}/siamese3_{}_{}'.format(arg.name, arg.hidden, arg.patch)
    device = torch.device('cpu') if arg.gpu == -1 else torch.device('cuda:{}'.format(arg.gpu))
    # 加载数据集
    m = loadmat('data/{0}/{0}.mat'.format(arg.name))
    data = m[config.get(arg.name, 'data_key')]
    data = data.astype(np.float)
    # 数据标准化
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data = scale(data)
    data = data.reshape((h, w, c))
    encoder = Net_Depthwise(1, arg.hidden, c)
    pair_vector_length = 2 * encoder.encoder.out_dim * arg.hidden
    metric_module = nn.Sequential(nn.Linear(pair_vector_length, pair_vector_length // 2),
                                  nn.ReLU(),
                                  nn.Linear(pair_vector_length // 2, pair_vector_length // 2),
                                  nn.ReLU(),
                                  nn.Linear(pair_vector_length // 2, 2))
    net = nn.Sequential(encoder, metric_module)
    # 训练训练10组数据
    for r in range(arg.run):
        # 重置模型参数
        encoder.apply(weight_init)
        metric_module.apply(weight_init)
        # 保存路径
        save_path = os.path.join(save_root_path, str(r))
        # 绘画loss, mse_loss和constrative_loss图
        viz.line([[0., 0., 0.]], [0], win='{} loss&acc {}'.format(arg.name, r),
                 opts={'title': '{} loss&acc {}'.format(arg.name, r),
                       'legend': ['train', 'test', 'accuracy']})
        viz.line([0.], [0], win='{} grad {}'.format(arg.name, r),
                 opts={'title': '{} grad {}'.format(arg.name, r)})
        # viz.line([0.], [0], win='{} accuracy {}'.format(dataset_name, r),
        #          opts={'title': '{} accuracy {}'.format(dataset_name, r)})
        print('*'*5 + 'RUN {}'.format(r) + '*'*5)
        # 读取训练样本和测试样本的标签
        m = loadmat('trainTestSplit/{}/sample{}_run{}.mat'.format(arg.name, arg.spc, r))
        train_gt, test_gt = m['train_gt'], m['test_gt']
        train_gt, test_gt = train_gt.astype(np.int32), test_gt.astype(np.int32)
        # 构造数据集
        train_dataset = HSIDatasetPair(data, train_gt, patch_size=arg.patch)
        test_dataset = HSIDatasetPair(data, test_gt, patch_size=arg.patch)
        if arg.name == 'gf5':
            length = len(test_dataset)
            size = int(0.6 * length)
            test_dataset, _ = random_split(test_dataset, [size, length - size])
        # 20%的数据作为验证集，加快模型的训练过程
        length = len(test_dataset)
        val_length = int(length * 0.2)
        test_dataset, _ = random_split(test_dataset, [val_length, length - val_length])
        train_loader = DataLoader(train_dataset, batch_size=10, num_workers=2, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=arg.batchsz, num_workers=4, pin_memory=True)

        # 构造二分类器，优化器，损失函数
        optimizer = optim.Adam(net.parameters())
        criterion = nn.CrossEntropyLoss()
        # 构造训练器
        trainer = PairTrainer(net)
        # 构造梯度监控器
        monitor = GradMonitor()
        # 训练模型
        max_acc = 0
        for epoch in range(arg.epoch):
            print('***** EPOCH: {} *****'.format(epoch))
            train_loss, grad = trainer.train(train_loader, optimizer, criterion,
                                             device, monitor)
            test_loss, acc = trainer.evaluate(test_loader, criterion, device)
            print('train loss: {} test_loss: {} acc: {}'.format(train_loss, test_loss, acc))
            # 绘画曲线图
            viz.line([[train_loss, test_loss, acc]], [epoch],
                     win='{} loss&acc {}'.format(arg.name, r), update='append')
            viz.line([grad], [epoch], win='{} grad {}'.format(arg.name, r), update='append')
            # viz.line([acc], [epoch], win='{} accuracy'.format(dataset_name), update='append')

            if acc > max_acc:
                max_acc = acc
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(encoder.state_dict(), os.path.join(save_path, 'best.pkl'))

        print('***** FINISH *****')





