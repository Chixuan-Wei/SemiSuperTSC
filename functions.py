import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pywt


# 加载数据
def loadDataset(dataset_path, dataset_name):
    train_file = os.path.join(dataset_path, dataset_name, dataset_name + "_TRAIN.tsv")
    test_file = os.path.join(dataset_path, dataset_name, dataset_name + "_TEST.tsv")

    train = pd.read_csv(train_file, sep="\t", header=None)
    test = pd.read_csv(test_file, sep="\t", header=None)

    data = pd.concat((train, test))  # 将训练集与测试集合并在一起

    labels = data.values[:, 0]  # 获取所有实例的标签
    num_classes = len(np.unique(labels))  # 类别的个数

    labels = (labels - labels.min()) / (labels.max() - labels.min()) * (num_classes - 1)  # 将类标签转换为0-num_classes
    data[data.columns[0]] = labels

    return data


# 划分数据集
def stratifiedSampling(data, seed, normalization=False, device='cpu'):
    train, test = train_test_split(data, test_size=0.2, random_state=seed, shuffle=True, stratify=data.values[:, 0])
    train, validate = train_test_split(train, test_size=0.1, random_state=seed, shuffle=True,
                                       stratify=train.values[:, 0])

    train_unlabeled, train_labeled = train_test_split(train, test_size=0.1, random_state=seed, shuffle=True,
                                                      stratify=train.values[:, 0])

    train_labeled_y = train_labeled.values[:, 0]
    train_labeled_x = train_labeled.values[:, 1:]

    train_unlabeled_x = train_unlabeled.values[:, 1:]

    validate_y = validate.values[:, 0]
    validate_x = validate.values[:, 1:]

    test_y = test.values[:, 0]
    test_x = test.values[:, 1:]

    # 数据标准化
    if normalization:
        train_x = np.vstack((train_labeled_x, train_unlabeled_x))
        mean = train_x.mean()
        std = train_x.std()

        train_labeled_x = (train_labeled_x - mean) / std
        train_unlabeled_x = (train_unlabeled_x - mean) / std
        validate_x = (validate_x - mean) / std
        test_x = (test_x - mean) / std

    train_labeled_x = train_labeled_x[:, np.newaxis, :]
    train_unlabeled_x = train_unlabeled_x[:, np.newaxis, :]
    validate_x = validate_x[:, np.newaxis, :]
    test_x = test_x[:, np.newaxis, :]

    # 将数据转移到指定设备
    train_labeled_x = torch.from_numpy(train_labeled_x).to(device)
    train_unlabeled_x = torch.from_numpy(train_unlabeled_x).to(device)
    validate_x = torch.from_numpy(validate_x).to(device)
    test_x = torch.from_numpy(test_x).to(device)
    train_labeled_y = torch.from_numpy(train_labeled_y).to(device)
    validate_y = torch.from_numpy(validate_y).to(device)
    test_y = torch.from_numpy(test_y).to(device)

    return train_labeled_x, train_labeled_y, train_unlabeled_x, validate_x, validate_y, test_x, test_y


# 获取子序列
def getPreAndPostSubsequences(X, length, stride):
    pre = []
    post = []
    for i in range(0, X.shape[2], stride):
        if (i + length * 2 <= X.shape[2]):
            pre.append(X[:, :, i:i + length])
            post.append(X[:, :, i + length:i + length * 2])

    pre = torch.cat(pre)
    post = torch.cat(post)
    return pre, post


# 小波变换
def waveletTransform(X, level=2):
    ans = []
    for i in range(0, X.shape[0], 1):
        coeffs = pywt.wavedec(X[i, :, :], 'db4', level=level)
        temp = []
        for coeff in coeffs:
            temp = np.append(temp, coeff)
        ans.append(temp)

    ans = torch.Tensor(ans)
    ans = ans[:, np.newaxis, :]
    return ans
