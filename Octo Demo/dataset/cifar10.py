import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
from PIL import Image
# dataset_root = "."
# batch_size = 256
#
# transform = transforms.Compose(
#     [
#         transforms.Resize(32),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ]
# )
#
# trainset = datasets.CIFAR10(root=dataset_root, train=True,
#                                  download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                                 shuffle=True, num_workers=0)
#
# testset = datasets.CIFAR10(root=dataset_root, train=False,
#                                 download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                                shuffle=False, num_workers=0)
#
# dataset = {}
#
# print(type(trainset), type(train_loader))

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 从源文件读取数据
# 返回 train_data[50000,3072]和labels[50000]
#    test_data[10000,3072]和labels[10000]
def get_data(train=False):
    data = None
    labels = None
    if train == True:
        for i in range(1, 6):
            batch = unpickle('../dataset/cifar-10-batches-py/data_batch_' + str(i))
            if i == 1:
                data = batch[b'data']
            else:
                data = np.concatenate([data, batch[b'data']])

            if i == 1:
                labels = batch[b'labels']
            else:
                labels = np.concatenate([labels, batch[b'labels']])
    else:
        batch = unpickle('../dataset/cifar-10-batches-py/test_batch')
        data = batch[b'data']
        labels = batch[b'labels']
    return np.array(data), np.array(labels)

def _convert_numpy():
    dataset = {}
    dataset['train_img'], dataset['train_label']  = get_data(train= True)
    dataset['test_img'], dataset['test_label'] = get_data(train=False)

    # print(dataset, type(dataset['train_img']))
    return dataset


def load_cifar10(normalize=True, flatten=True, one_hot_label=False):
    """读入cifar10数据集

    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组

    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """

    dataset = _convert_numpy()
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0


    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 3, 32, 32)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])



# 图像预处理函数，Compose会将多个transform操作包在一起
# 对于彩色图像，色彩通道不存在平稳特性
transform = transforms.Compose([
    # ToTensor是指把PIL.Image(RGB) 或者numpy.ndarray(H x W x C)
    # 从0到255的值映射到0到1的范围内，并转化成Tensor格式。
    transforms.ToTensor(),
    # Normalize函数将图像数据归一化到[-1,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
)


# 将标签转换为torch.LongTensor
def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


'''
自定义数据集读取框架来载入cifar10数据集
需要继承data.Dataset
'''

#
# class Cifar10_Dataset(Data.Dataset):
#     def __init__(self, train=True, transform=None, target_transform=None):
#         # 初始化文件路径
#         self.transform = transform
#         self.target_transform = target_transform
#         self.train = train
#         # 载入训练数据集
#         if self.train:
#             self.train_data, self.train_labels = get_data(train)
#             self.train_data = self.train_data.reshape((50000, 3, 32, 32))
#             # 将图像数据格式转换为[height,width,channels]方便预处理
#             self.train_data = self.train_data.transpose((0, 2, 3, 1))
#             # 载入测试数据集
#         else:
#             self.test_data, self.test_labels = get_data()
#             self.test_data = self.test_data.reshape((10000, 3, 32, 32))
#             self.test_data = self.test_data.transpose((0, 2, 3, 1))
#         pass
#
#     def __getitem__(self, index):
#         # 从数据集中读取一个数据并对数据进行
#         # 预处理返回一个数据对，如（data,label）
#         if self.train:
#             img, label = self.train_data[index], self.train_labels[index]
#         else:
#             img, label = self.test_data[index], self.test_labels[index]
#
#         img = Image.fromarray(img)
#         # 图像预处理
#         if self.transform is not None:
#             img = self.transform(img)
#         # 标签预处理
#         if self.target_transform is not None:
#             target = self.target_transform(label)
#
#         return img, target
#
#     def __len__(self):
#         # 返回数据集的size
#         if self.train:
#             return len(self.train_data)
#         else:
#             return len(self.test_data)


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_cifar10()
    print( type(x_train))