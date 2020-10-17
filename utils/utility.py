import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torch.distributions import Normal
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from numpy import genfromtxt, mean, std
import numpy as np
import os
import subprocess
from pprint import pprint


def generate_data(n_data, n_test, n_dim):
    device = get_cuda_device()
    data = dict()
    test = dict()
    A = ts(np.random.random((n_dim, 1))).to(device)
    data['X'] = ts(np.random.random((n_data, n_dim))).to(device)
    data['Y'] = torch.mm(data['X'], A)
    test['X'] = ts(np.random.random((n_test, n_dim))).to(device)
    test['Y'] = torch.mm(test['X'], A)
    return data, test


def abalone_data(is_train = True):
    tail = 'train' if is_train else 'test'
    datapath = './data/abalone/abalone.{}'.format(tail)
    device = get_cuda_device()
    X, y = [], []

    with open(datapath) as f:
        for line in f:
            line = line.strip().split(',')
            # convert a line to numbers
            if line[0] == 'M': line[0] = 1
            elif line[0] == 'F': line[0] = 2
            else: line[0] = 3
            data = [float(x) for x in line[:-1]]
            target = float(line[-1])
            target = target - 1 if target != 29 else (target - 2) # index form 0
            X.append(data)
            y.append(target)

    X = np.array(X)
    y = np.array(y)
    y = y.reshape(len(y), 1)

    x_tensor = torch.from_numpy(X).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    xmean = torch.mean(x_tensor, dim = 0)
    xstd = torch.std(x_tensor, dim = 0)
    x_tensor = (x_tensor - xmean) / xstd
    return {'X': x_tensor.float(), 'Y': y_tensor.float()}, len(X)


def diabetes_data(is_train=True, n_train=392, n_test=50):
    device = get_cuda_device()
    x, y = datasets.load_diabetes(return_X_y=True)

    x = np.split(x, [n_train, n_train + n_test])
    y = np.split(y, [n_train, n_train + n_test])

    data = dict()
    data['dim'] = x[0].shape[1]
    data['train'] = {'X': ts(x[0]).float().to(device), 'Y': ts(y[0].reshape(-1, 1)).float().to(device)}
    data['test'] = {'X': ts(x[1]).float().to(device), 'Y': ts(y[1].reshape(-1, 1)).float().to(device)}

    if is_train:
        return data['train'], n_train
    else:
        return data['test'], n_test


def gas_sensor_data(datapath = './data/gas-sensor', pct_train = 0.8, is_preload = True):
    if not is_preload:
        file_list = [os.path.join(datapath, x)
                     for x in os.listdir(datapath) if x != 'README.txt']
        print(file_list)
        data_list = [genfromtxt(l, skip_header = 1, delimiter = ',') for l in file_list]
        normed_data = np.vstack(data_list)
        # skip first col
        normed_data = normed_data[:, 1:]
        # normalization - column based
        normed_data = (normed_data - mean(normed_data, 0)) / std(normed_data, 0)

        # split train and test
        len_data = len(normed_data)
        test_idx = int(len_data * pct_train)

        data_np = normed_data[:test_idx]
        test_np = normed_data[test_idx:]

        np.save(os.path.join(datapath, 'data_np.npy'), data_np)
        np.save(os.path.join(datapath, 'test_np.npy'), test_np)
    else:
        data_np = np.load(os.path.join(datapath, 'data_np.npy'))
        test_np = np.load(os.path.join(datapath, 'test_np.npy'))

    data, test = {}, {}
    device = get_cuda_device()
    data['X'] = torch.from_numpy(data_np[:, 1:]).to(device).float()
    data['Y'] = torch.from_numpy(data_np[:, 0]).reshape(-1, 1).to(device).float()
    test['X'] = torch.from_numpy(test_np[:, 1:]).to(device).float()
    test['Y'] = torch.from_numpy(test_np[:, 0]).reshape(-1, 1).to(device).float()

    return data, test


def sample_rows(prob, n_rows):
    selected = []
    for i in range(n_rows):
        p = np.random.random()
        if p < prob[i]:
            selected.append(i)
    return selected


def rmse(Y1, Y2):
    diff = Y1.float() - Y2.float()
    return torch.sqrt(torch.dot(diff.view(-1), diff.view(-1)) / diff.shape[0])


def nll_test(pred, var, truth):
    m = torch.distributions.MultivariateNormal(pred.view(-1), var)
    return -1.0 * m.log_prob(truth.view(-1))


def ts(X):
    return torch.tensor(X)


def dt(X):
    return X.detach().cpu().numpy()


def get_cuda_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_seed = seed
    np_seed = seed
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    pprint(gpu_memory_map)


if __name__ == "__main__":
    train, test = gas_sensor_data(is_preload = True)
    print(train['X'].shape, test['X'].shape)