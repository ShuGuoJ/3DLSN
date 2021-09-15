'''测试孪生网络的分类准确率'''
import torch
from torch import nn, optim
import numpy as np
from scipy.io import loadmat
from utils import weight_init, train_test_split
from HSIDataset import HSIDatasetPair, HSIDataset
from Model.module import Net_Depthwise
from torch.utils.data import DataLoader
import os
import argparse
from visdom import Visdom
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os
from matplotlib import pyplot as plt
from configparser import ConfigParser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='DATASET NAME')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID')
    parser.add_argument('--run', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--patch', type=int, default=7)
    parser.add_argument('--spc', type=int, default=10)
    parser.add_argument('--batchsz', type=int, default=128)
    arg = parser.parse_args()
    config = ConfigParser()
    config.read('dataInfo.ini')
    device = torch.device('cpu') if arg.gpu == -1 else torch.device('cuda:{}'.format(arg.gpu))
    # 加载数据集
    m = loadmat('data/{0}/{0}.mat'.format(arg.name))
    data = m[config.get(arg.name, 'data_key')]
    # m = loadmat('data/{}/PaviaU_gt.mat')
    # gt = m[info['label_key']]
    data = data.astype(np.float)
    # 数据标准化
    h, w, c = data.shape
    data = data.reshape((h*w, c))
    data = scale(data)
    data = data.reshape((h, w, c))
    # # gt += 1
    # train_gt, test_gt = train_test_split(gt, 10)
    # # test_gt, _ = train_test_split(test_gt, 200)
    # 构造数据集
    print('*'*5 + arg.name + '*'*5)
    for r in range(arg.run):
        # 模型保存路径
        save_path = 'depthwise_models/{}/siamese3_{}_{}/{}'.format(arg.name, arg.hidden, arg.patch, r)
        # 读取训练样本和测试样本的标签
        m = loadmat('trainTestSplit/{}/sample{}_run{}.mat'.format(arg.name, arg.spc, r))
        train_gt, test_gt = m['train_gt'], m['test_gt']
        # data, gt = data.astype(np.float), gt.astype(np.int32)
        train_gt, test_gt = train_gt.astype(np.int32), test_gt.astype(np.int32)
        train_dataset = HSIDataset(data, train_gt, patch_size=arg.patch)
        test_dataset = HSIDataset(data, test_gt, patch_size=arg.patch)
        train_loader = DataLoader(train_dataset, batch_size=10, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=arg.batchsz, num_workers=4)

        # files = os.listdir(SAVE_PATH)
        # net = Net_Depthwise(1, 128)
        net = Net_Depthwise(1, arg.hidden, c)
        net.load_state_dict(torch.load(os.path.join(save_path, 'best.pkl')))
        net.eval()
        net.to(device)

        # classifier = SVC(random_state=666)
        classifier = LogisticRegression(random_state=666, max_iter=4000)
        data_train, y_train = [], []
        # 训练
        for x, target in train_loader:
            x = x.to(device)
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
            with torch.no_grad():
                features = net(x)
            data_train.append(features.clone().detach())
            y_train.append(target.clone().detach())
        data_train = torch.cat(data_train, dim=0)
        data_train = data_train.reshape((data_train.size(0), -1))

        y_train = torch.cat(y_train, dim=0)
        data_train, y_train = data_train.cpu().numpy(), y_train.cpu().numpy()
        classifier.fit(data_train, y_train)
        # 测试
        data_test = []
        y_test_true = []
        for x, target in test_loader:
            x = x.to(device)
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
            with torch.no_grad():
                features = net(x)
            data_test.append(features.clone().detach())
            y_test_true.append(target.clone().detach())
        data_test = torch.cat(data_test, dim=0)
        data_test = data_test.reshape((data_test.size(0), -1))
        y_test_true = torch.cat(y_test_true, dim=0)
        data_test, y_test_true = data_test.cpu().numpy(), y_test_true.cpu().numpy()
        y_test_pred = classifier.predict(data_test)
        score = classifier.score(data_test, y_test_true)
        joblib.dump(classifier, os.path.join(save_path, 'classifier.pkl'))

        # plt.plot(scores)
        # plt.savefig(os.path.join(save_path, 'accuracy.jpg'))




