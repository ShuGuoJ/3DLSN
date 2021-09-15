import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA
import random
from sklearn.preprocessing import scale


class HSIDataset(Dataset):
    def __init__(self, hsi, gt, patch_size=1):
        '''
        :param hsi: [h, w, bands]
        :param gt: [h, w]
        :param patch_size: scale
        '''
        super(HSIDataset, self).__init__()
        self.hsi = self.add_mirror(hsi, patch_size)  # [h, w, bands]
        self.gt = gt  # [h, w]
        self.patch_size = patch_size
        # 标签数据的索引
        self.indices = tuple(zip(*np.nonzero(gt)))

    # 添加镜像
    @staticmethod
    def add_mirror(data, patch_size):
        dx = patch_size // 2
        if dx != 0:
            h, w, c = data.shape
            mirror = np.zeros((h + 2 * dx, w + 2 * dx, c))
            mirror[dx:-dx, dx:-dx, :] = data
            for i in range(dx):
                # 填充左上部分镜像
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                # 填充右下部分镜像
                mirror[:, -i - 1, :] = mirror[:, -(2 * dx - i) - 1, :]
                mirror[-i - 1, :, :] = mirror[-(2 * dx - i) - 1, :, :]
        else:
            mirror = data
        return mirror

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        '''
        :param index:
        :return: 元素光谱信息， 元素的空间信息， 标签
        '''
        x, y = self.indices[index]
        # 领域: [patchsz, patchsz, bands]
        neighbor_region = self.hsi[x: x + self.patch_size, y: y + self.patch_size, :]
        # 类别
        target = self.gt[x, y] - 1
        return torch.tensor(neighbor_region, dtype=torch.float), torch.tensor(target, dtype=torch.long)


class HSIDatasetPair(HSIDataset):
    def __init__(self, hsi, gt, patch_size=1):
        super().__init__(hsi, gt, patch_size)
        nc = self.gt.max()
        non_zero = gt != 0
        self.pos_pool = [tuple(zip(*np.nonzero(gt == i))) for i in range(1, nc + 1)]
        self.neg_pool = [tuple(zip(*np.nonzero((gt != i) & non_zero))) for i in range(1, nc + 1)]

    def __getitem__(self, index):

        x, y = self.indices[index]
        target = self.gt[x, y]

        x_pos, y_pos = random.sample(self.pos_pool[target - 1], 1)[0]

        x_neg, y_neg = random.sample(self.neg_pool[target - 1], 1)[0]

        neighbor_region = self.hsi[x: x + self.patch_size, y: y + self.patch_size, :]
        neighbor_region_pos = self.hsi[x_pos: x_pos + self.patch_size, y_pos: y_pos + self.patch_size, :]
        neighbor_region_neg = self.hsi[x_neg: x_neg + self.patch_size, y_neg: y_neg + self.patch_size, :]

        pos_sample = np.stack([neighbor_region, neighbor_region_pos], axis=0)
        neg_sample = np.stack([neighbor_region, neighbor_region_neg], axis=0)

        sample = np.stack([pos_sample, neg_sample], axis=0)
        return torch.tensor(sample, dtype=torch.float), torch.tensor([True, False], dtype=torch.long)


