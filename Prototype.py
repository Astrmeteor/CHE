import tenseal.enc_context
import torch
import torchvision
from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import tenseal as ts
import time
from tqdm import tqdm
from seal import *
import math
import scipy.stats
import os
from multiprocessing import cpu_count
from itertools import product
import threading
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from time import ctime
import multiprocessing
import pathos
import functools

import re
import _ctypes
import os
import copyreg

# model
class LeNet(torch.nn.Module):
    def __init__(self, input_channel=3, output=10):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(400, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, output)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv1_act = conv1.pow(2)
        # conv1_act = F.relu(conv1)

        # conv1_avg = F.max_pool2d(conv1_act, 2)
        conv1_avg = F.avg_pool2d(conv1_act, 2)


        conv2 = self.conv2(conv1_avg)
        conv2_act = conv2.pow(2)
        # conv2_act = F.relu(conv2)
        # conv2_avg = F.max_pool2d(conv2_act, 2)
        conv2_avg = F.avg_pool2d(conv2_act, 2)

        x_pac = conv2_avg.view(conv2_avg.size()[0], -1)

        fc1 = self.fc1(x_pac)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        return fc3

class TenSEAL_ConvNet(torch.nn.Module):
    def __init__(self, hidden = 16, output =10):
        super(TenSEAL_ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.conv2 = torch.nn.Conv2d(4, 8, kernel_size=3, padding=0, stride=1)
        self.fc1 = torch.nn.Linear(32, hidden) # 64
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        conv1 = self.conv1(x)
        act = conv1.pow(2)
        avg = F.avg_pool2d(act, 2)
        conv2 = self.conv2(avg)
        # act = conv1.pow(2)
        # avg1 = F.avg_pool2d(act, 2)
        flatten = conv2.view(-1, 32) # 64
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)

        return fc2

class MNIST_BN_CNN(torch.nn.Module):
    def __init__(self, input_channel = 1, output = 10):
        super(MNIST_BN_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, 5, kernel_size = 5, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(num_features=5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(5, 50, kernel_size = 5, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc = torch.nn.Linear(800, 10)

    def forward(self, x):
        # conv-bn-act
        conv1 = self.conv1(x)
        
        bn1 = self.bn1(conv1)
        # act1 = F.relu(bn1)
        act1 = bn1.pow(2)
        '''
        # conv-act-bn
        act1 = conv1.pow(2)
        bn1 = self.bn1(act1)
        # act1 = F.relu(bn1)
        '''
        
        # conv-bn-act
        conv2 = self.conv2(act1)
        
        bn2 = self.bn2(conv2)
        # act2 = F.relu(bn2)
        act2 = bn2.pow(2)
        '''
        # conv-act-bn
        act2 = conv2.pow(2)
        bn2 = self.bn2(act2)
        # act2 = F.relu(bn2)
        '''

        x_pac = act2.view(act2.size()[0], -1)
        # x_pac = bn2.view(bn2.size()[0], -1)

        fc = self.fc(x_pac)

        return fc

def test(model, test_loader, criterion):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    correct = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()
    for data, target in test_loader:

        output = model(data)
        test_loss += criterion(output, target)
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set loss: {} ; Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

def train(net, train_loader, optimizer, criterion):
    # training
    epochs = 100
    net.train()
    device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")
    print(torch.get_num_threads())
    for epoch in range(epochs):
        print("Training Epoch {}".format(epoch))
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            loss_ = loss.mean().item()
            optimizer.step()
        print("Loss {:.8f}".format(loss_))

    # PATH = f'Parameters/Alter_cifar10_{net.__class__.__name__}_{epochs}.pth'
    PATH = f'Parameters/mnist_{net.__class__.__name__}_{epochs}.pth'
    torch.save(net.state_dict(), PATH)

    return net

def vec_convolution(parms, idx, x, y):

    v1 = np.vectorize(vec1)
    v1.excluded.add(0)
    # v1.excluded.add(1)
    v1.excluded.add(2)
    Z = []
    for i in range(y.shape[0]):
        Z.append(
            parms['evaluator'].add_many(
                v1(parms, idx, x[i], y[i]).flatten().tolist()
            ))
    Z = parms['evaluator'].add_many(Z)
    return Z
def vec1(parms, idx, x, y):
    weight = y
    temp = parms['evaluator'].rotate_vector(x, idx, parms['galois_keys'])
    parms['evaluator'].mod_switch_to_inplace(weight, x.parms_id())
    z = parms['evaluator'].multiply_plain(temp, weight)
    parms['evaluator'].rescale_to_next_inplace(z)
    return z

class EncBN_CNN:
    def __init__(self, torch_nn, parms):
        self.parms = parms

        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.in_channels,
            torch_nn.conv1.kernel_size[0] * torch_nn.conv1.kernel_size[1]
        )  # out, in, fxf
        self.conv1_weight = self.weight_change(self.conv1_weight)
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        self.conv1_bias = self.bias_change(self.conv1_bias)
        self.conv1_stride = torch_nn.conv1.stride[0]

        self.conv2_weight = torch_nn.conv2.weight.data.view(
            torch_nn.conv2.out_channels, torch_nn.conv2.in_channels,
            torch_nn.conv2.kernel_size[0] * torch_nn.conv2.kernel_size[1]
        )  # out, in, fxf
        self.conv2_weight = self.weight_change(self.conv2_weight)
        self.conv2_bias = torch_nn.conv2.bias.data.tolist()
        self.conv2_bias = self.bias_change(self.conv2_bias)
        self.conv2_stride = torch_nn.conv2.stride[0]

        self.bn1_mu = torch_nn.bn1.running_mean.data.numpy()
        self.bn1_sigma = torch_nn.bn1.running_var.data.numpy()
        self.bn1_gamma = torch_nn.bn1.weight.data.numpy()
        self.bn1_beta = torch_nn.bn1.bias.data.numpy()
        self.bn1_eps = torch_nn.bn1.eps
        self.bn1_a, self.bn1_b, self.bn1_c = self.coefficients(self.bn1_mu, self.bn1_sigma, self.bn1_gamma,
                                                               self.bn1_beta, self.bn1_eps, 'CBA')
        self.conv2_weight = self.conv2_weight * torch.tensor(self.bn1_a).reshape(self.bn1_a.size, 1)

        self.bn2_mu = torch_nn.bn2.running_mean.data.numpy()
        self.bn2_sigma = torch_nn.bn2.running_var.data.numpy()
        self.bn2_gamma = torch_nn.bn2.weight.data.numpy()
        self.bn2_beta = torch_nn.bn2.bias.data.numpy()
        self.bn2_eps = torch_nn.bn2.eps
        self.bn2_a, self.bn2_b, self.bn2_c = self.coefficients(self.bn2_mu, self.bn2_sigma, self.bn2_gamma,
                                                               self.bn2_beta, self.bn2_eps, "CBA")
        self.conv3_weight = self.conv3_weight * torch.tensor(self.bn2_a).reshape(self.bn2_a.size, 1)

        self.bn3_mu = torch_nn.bn3.running_mean.data.numpy()
        self.bn3_sigma = torch_nn.bn3.running_var.data.numpy()
        self.bn3_gamma = torch_nn.bn3.weight.data.numpy()
        self.bn3_beta = torch_nn.bn3.bias.data.numpy()
        self.bn3_eps = torch_nn.bn3.eps
        self.bn3_a, self.bn3_b, self.bn3_c = self.coefficients(self.bn3_mu, self.bn3_sigma, self.bn3_gamma,
                                                               self.bn3_beta, self.bn3_eps, "CBA")

        self.fc1_weight = torch_nn.fc1.weight.data.T.tolist()
        self.fc1_weight = self.weight_diag(self.fc1_weight)
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.data.T.tolist()
        self.fc2_weight = self.weight_diag(self.fc2_weight)
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

        self.torch_nn = torch_nn
        # valid vector
        self.valid_vector = 0

    def count_rotation(self, m, f):
        """
        :param m: input size
        :param f: filter size
        :param p: padding number
        :return: rotation index
        """
        idx = []
        for i in range(f):
            start = i * m
            for j in range(f):
                if i == f - 1 & j == f - 1:
                    continue
                a = start + j
                if j == f - 1:
                    a = (i + 1) * m - 1
                    idx.append(a + 1)
                else:
                    idx.append(a + 1)
        assert len(idx) == pow(f, 2) - 1
        return idx

    def weight_change(self, weight):

        weight = weight.numpy()
        weight_temp = np.zeros(weight.shape).tolist()
        for outer, inner, i in product(range(weight.shape[0]), range(weight.shape[1]), range(weight.shape[2])):
            weight_temp[outer][inner][i] = self.parms['encoder'].encode(weight[outer][inner][i].item(), self.parms['scale'])

        return  weight_temp

    def bias_change(self, bias):
        for i in range(bias.__len__()):
            bias[i] = self.parms['encoder'].encode(bias[i], self.parms['scale'])
        return bias

    def mask_computation(self, x, n, valid_vector, rot_index):
        """
        :param x: input  HE encrypted vector
        :param n: side of output
        :param f: side of filter
        :param g: gap
        :param N: polynomial modulus degree
        :return: well-structure vector
        """
        y = x
        # idx: valuable idx; rot_idx: rotation index
        idx = valid_vector
        rot_idx = rot_index
        # generate all-zeroes mask
        mask = [0 for h in range(self.parms['slots'])]
        mask[0] = 1
        mask = self.parms['encoder'].encode(mask, self.parms['scale'])
        self.parms['evaluator'].mod_switch_to_inplace(mask, y.parms_id())
        Z = []
        Z.append(self.parms['evaluator'].multiply_plain(y, mask))

        for k in range(self.parms['n'] * self.parms['n'] - 1):
        # HE element-wise multiplication and addition
            mask = [0 for h in range(self.parms['slots'])]
            mask[idx[k + 1]] = 1
            mask = self.parms['encoder'].encode(mask, self.parms['scale'])
            self.parms['evaluator'].mod_switch_to_inplace(mask, y.parms_id())
            Z.append(
                self.parms['evaluator'].rotate_vector(
                    self.parms['evaluator'].multiply_plain(y, mask),
                    rot_idx[k], self.parms['galois_keys']
                )
            )
        Z = self.parms['evaluator'].add_many(Z)

        return Z

    def vec_conv(self, input, conv_weight, conv_bias, s):
        out_channels = []
        in_channel = conv_weight[0].__len__()
        out_channel = conv_weight.__len__()
        f = int(math.sqrt(conv_weight[0][0].__len__()))
        p = 0
        self.parms['n'] = int((self.parms['m'] + 2 * p - f) / s + 1)

        # search rotation index
        idx = self.count_rotation(self.parms['m'], f)
        for i in range(idx.__len__()):
            idx[i] = self.valid_vector[idx[i]]
        idx.insert(0,0)

        # valid values
        valid_index = []
        mask = [0 for i in range(self.parms['slots'])]
        for i in range(self.parms['n']):
            for j in range(self.parms['n']):
                valid_index.append(i * s * self.parms['m'] + j * s)
                mask[self.valid_vector[i * s * self.parms['m'] + j * s]] = 1
        mask = self.parms['encoder'].encode(mask, self.parms['scale'])
        Y = []
        Z = []
        # input -> numpy data
        input = np.array(input)
        conv_weight = np.array(conv_weight)# .transpose((1,2,0))
        for i in range(conv_weight.shape[0]):
            Y.append(vec_convolution(self.parms, idx, input, conv_weight[i]))

        for i in range(out_channels.__len__()):
            # add bias
            bias = conv_bias[i]
            Z = out_channels[i]
            self.parms['evaluator'].mod_switch_to_inplace(bias, Z.parms_id())
            Z.scale(2 ** int(math.log2(bias.scale())))
            self.parms['evaluator'].add_plain_inplace(Z, bias)
            mask_encode = mask
            self.parms['evaluator'].mod_switch_to_inplace(mask_encode, Z.parms_id())
            self.parms['evaluator'].multiply_plain_inplace(Z, mask_encode)
            self.parms['evaluator'].rescale_to_next_inplace(Z)
            out_channels[i] = Z


        # set output of layer as input of next layer
        self.parms['m'] = self.parms['n']
        # set invalid vector
        Z = []
        for i in valid_index:
            Z.append(self.valid_vector[i])
        self.valid_vector = Z

        return out_channels

    def convolution(self, input, conv_weight, conv_bias, s):
        """
        :param input: vector include encrypted data: should be squeezed for simplicity
        :param conv_weight: convolution layer weight
        :param conv_bias: convolution layer bias
        :param m: side of input
        :return: after convolution computation
        """
        out_channels = []
        in_channel = conv_weight[0].__len__()
        out_channel = conv_weight.__len__()
        f = int(math.sqrt(conv_weight[0][0].__len__()))
        p = 0
        self.parms['n'] = int((self.parms['m'] + 2 * p - f) / s + 1)

        # search rotation index
        idx = self.count_rotation(self.parms['m'], f)
        for i in range(idx.__len__()):
            idx[i] = self.valid_vector[idx[i]]

        # valid values
        valid_index = []
        mask = [0 for i in range(self.parms['slots'])]
        for i in range(self.parms['n']):
            for j in range(self.parms['n']):
                valid_index.append(i * s * self.parms['m'] + j * s)
                mask[self.valid_vector[i * s * self.parms['m'] + j * s]] = 1
        mask = self.parms['encoder'].encode(mask, self.parms['scale'])

        inputs = [[0 for i in range(f * f)] for j in range(in_channel)]

        for i in range(in_channel):
            inputs[i][0] = input[i]
            for j in range(f * f):
                inputs[i][j] = self.parms['evaluator'].rotate_vector(input[i], idx[j - 1], self.parms['galois_keys'])

        for outer in range(out_channel):
            Z = []
            for inner in range(in_channel):
                Y = []
                for i in range(f * f):
                    self.parms['evaluator'].mod_switch_to_inplace(conv_weight[outer][inner][i],
                                                                  inputs[inner][i].parms_id())
                    Y.append(self.parms['evaluator'].multiply_plain(
                        inputs[inner][i], conv_weight[outer][inner][i]))
                Z.append(self.parms['evaluator'].add_many(Y))
            Z = self.parms['evaluator'].add_many(Z)
            self.parms['evaluator'].rescale_to_next_inplace(Z)
            out_channels.append(Z)

        for i in range(out_channels.__len__()):
            # add bias
            bias = conv_bias[i]
            Z = out_channels[i]
            self.parms['evaluator'].mod_switch_to_inplace(bias, Z.parms_id())
            Z.scale(2 ** int(math.log2(bias.scale())))
            self.parms['evaluator'].add_plain_inplace(Z, bias)
            mask_encode = mask
            self.parms['evaluator'].mod_switch_to_inplace(mask_encode, Z.parms_id())
            self.parms['evaluator'].multiply_plain_inplace(Z, mask_encode)
            self.parms['evaluator'].rescale_to_next_inplace(Z)
            out_channels[i] = Z

        # set output of layer as input of next layer
        self.parms['m'] = self.parms['n']
        # set invalid vector
        Z = []
        for i in valid_index:
            Z.append(self.valid_vector[i])
        self.valid_vector = Z

        return out_channels

    def activation(self, x, name):
        channels = len(x)
        for i in range(channels):
            if name == 'square':
                # HE square
                self.parms['evaluator'].square_inplace(x[i])
                self.parms['evaluator'].relinearize_inplace(x[i], self.parms['relin_keys'])
                self.parms['evaluator'].rescale_to_next_inplace(x[i])
        return x

    def coefficients(self, mu, sigma, gamma, beta, eps, name):
        if name == 'CBA':
            temp = gamma / np.sqrt(sigma + eps)
            a = np.power(temp, 2)
            b = 2 * (beta - temp * mu) * temp
            c = np.power((beta - temp * mu), 2)
        elif name == 'CAB':
            a = gamma / np.sqrt(sigma + eps)
            b = 0
            c = beta - a * mu
        return a, b, c

    def bn_act(self, x, a, b, c):
        channels = len(x)
        X = []
        for i in range(channels):
                # HE square
                # a*x^2 -> a_x_2
                x_2 = self.parms['evaluator'].square(x[i])
                self.parms['evaluator'].relinearize_inplace(x_2, self.parms['relin_keys'])
                self.parms['evaluator'].rescale_to_next_inplace(x_2)
                a_prime = self.parms['encoder'].encode(a[i].item(), self.parms['scale'])
                self.parms['evaluator'].mod_switch_to_inplace(a_prime, x_2.parms_id())
                a_x_2 = self.parms['evaluator'].multiply_plain(x_2, a_prime)
                self.parms['evaluator'].rescale_to_next_inplace(a_x_2)

                # b * x
                b_prime = self.parms['encoder'].encode(b[i].item(), self.parms['scale'])
                self.parms['evaluator'].mod_switch_to_inplace(b_prime, x[i].parms_id())
                b_x = self.parms['evaluator'].multiply_plain(x[i], b_prime)
                self.parms['evaluator'].rescale_to_next_inplace(b_x)

                c_prime = self.parms['encoder'].encode(c[i].item(), self.parms['scale'])

                self.parms['evaluator'].mod_switch_to_inplace(c_prime, a_x_2.parms_id())
                self.parms['evaluator'].mod_switch_to_inplace(b_x, a_x_2.parms_id())
                a_x_2.scale(2 ** int(math.log2(c_prime.scale())))
                b_x.scale(2 ** int(math.log2(c_prime.scale())))

                X.append(self.parms['evaluator'].add_plain(self.parms['evaluator'].add(a_x_2, b_x), c_prime))
        return X

    def start_index(self, m, n, s):
        idx = []
        rot_idx = []
        for i in range(n):
            for j in range(n):
                idx.append(i * s * m + j * s)
        for j in range(1, n * n):
            rot_idx.append(idx[j] - j)
        # idx raw valid index; rot_idx rotation index
        return idx, rot_idx

    def average_pooling(self, x, f, s):
        '''
        :param x: input
        :param f: size of kernel
        :param s: stride
        :return:
        '''
        out_channel = x.__len__()
        self.parms['n'] = int((self.parms['m'] - f) / s + 1)
        out_channels = []

        # search rotation index
        idx = self.count_rotation(self.parms['m'], f)
        for i in range(idx.__len__()):
            idx[i] = self.valid_vector[idx[i]]

        # valid values
        valid_index = []
        mask = [0 for i in range(self.parms['slots'])]
        for i in range(self.parms['n']):
            for j in range(self.parms['n']):
                valid_index.append(i * s * self.parms['m'] + j * s)
                mask[self.valid_vector[i * s * self.parms['m'] + j * s]] = 1/(f*f)
        mask = self.parms['encoder'].encode(mask, self.parms['scale'])
        self.parms['evaluator'].mod_switch_to_inplace(mask, x[0].parms_id())

        for i in range(out_channel):
            # y = x[i]
            # rotate by raw
            Y = []
            Y.append(x[i])
            for j in range(1, f*f):
                # HE rotation and addition
                Y.append(self.parms['evaluator'].rotate_vector(x[i], idx[j-1], self.parms['galois_keys']))
            y = self.parms['evaluator'].add_many(Y)
            y = self.parms['evaluator'].multiply_plain(y, mask)
            self.parms['evaluator'].rescale_to_next_inplace(y)
            out_channels.append(y)

        # set invalid vector
        Z = []
        for i in valid_index:
            Z.append(self.valid_vector[i])
        self.valid_vector = Z
        self.parms['m'] = self.parms['n']

        return out_channels

    def pack_vectors(self, x, n):
        channels = x.__len__()
        Y = []
        # Y.append(x[0])
        rot_idx = []
        for j in range(1, n * n):
            rot_idx.append(self.valid_vector[j] - j)

        for i in range(channels):
            # HE rotation
            y = x[i]
            # idx: valuable idx; rot_idx: rotation index
            idx = self.valid_vector
            # generate all-zeroes mask
            mask = [0 for h in range(self.parms['slots'])]
            mask[0] = 1
            mask = self.parms['encoder'].encode(mask, self.parms['scale'])
            self.parms['evaluator'].mod_switch_to_inplace(mask, y.parms_id())
            Z = []
            Z.append(self.parms['evaluator'].multiply_plain(y, mask))
            for k in range(self.parms['n'] * self.parms['n'] - 1):
                # HE element-wise multiplication and addition
                mask = [0 for h in range(self.parms['slots'])]
                mask[idx[k + 1]] = 1
                mask = self.parms['encoder'].encode(mask, self.parms['scale'])
                self.parms['evaluator'].mod_switch_to_inplace(mask, y.parms_id())
                Z.append(
                    self.parms['evaluator'].rotate_vector(
                        self.parms['evaluator'].multiply_plain(y, mask),
                        rot_idx[k], self.parms['galois_keys']
                    )
                )
            y = self.parms['evaluator'].add_many(Z)
            if i == 0:
                Y.append(y)
            else:
                Y.append(self.parms['evaluator'].rotate_vector(y, -(n * n * i), self.parms['galois_keys']))

        Y = self.parms['evaluator'].add_many(Y)
        self.parms['evaluator'].rescale_to_next_inplace(Y)
        return Y

    def weight_diag(self, weight):
        weight = np.array(weight)
        weight = np.pad(weight, pad_width=((0, 0), (0, weight.shape[0] - weight.shape[1])))
        W = []
        for i in range(weight.shape[0]):
            if i == 0:
                W.append(np.diag(weight, i).tolist())
            else:
                W.append(np.concatenate([np.diag(weight, -i), np.diag(weight, weight.shape[0] - i)]).tolist())
        return W

    def fully_connected(self, x, weight, bias):
        Y = []
        temp = x

        rot_single = weight.__len__() - bias.__len__() + 1

        for i in range(weight.__len__()):

            w = self.parms['encoder'].encode(weight[i], self.parms['scale'])
            self.parms['evaluator'].mod_switch_to_inplace(w, x.parms_id())
            Y.append(
                self.parms['evaluator'].multiply_plain(temp, w)
            )
            if i != 0 or i < rot_single:
                temp = self.parms['evaluator'].rotate_vector(x, i + 1, self.parms['galois_keys'])

            B = []
            if i != weight.__len__() - 1 and i >= rot_single:
                B.append(self.parms['evaluator'].rotate_vector(x, i + 1, self.parms['galois_keys']))
                B.append(self.parms['evaluator'].rotate_vector(x, -weight.__len__() + i + 1, self.parms['galois_keys']))
                temp = self.parms['evaluator'].add_many(B)
        Y = self.parms['evaluator'].add_many(Y)
        self.parms['evaluator'].rescale_to_next_inplace(Y)

        B = self.parms['encoder'].encode(bias, self.parms['scale'])

        self.parms['evaluator'].mod_switch_to_inplace(B, Y.parms_id())
        Y.scale(2 ** int(math.log2(B.scale())))
        return self.parms['evaluator'].add_plain(Y, B)

    def forward(self,enc_x, x):  # try to reduce paramteres for functions
        start = time.time()
        self.parms['m'] = 28
        self.valid_vector = [i for i in range(self.parms['m'] * self.parms['m'])]

        # Conv1 MD 2
        conv1_convolution_start_time = time.time()
        conv1 = self.convolution(enc_x, self.conv1_weight, self.conv1_bias, self.conv1_stride)  # try to built-in m
        # conv1 = self.vec_conv(enc_x, self.conv1_weight, self.conv1_bias, self.conv1_stride)
        conv1_convolution_end_time = time.time()
        conv1_convolution_time = conv1_convolution_end_time - conv1_convolution_start_time
        print(f'Conv1 Convolution time: {conv1_convolution_time:.4f} s')

        # BN + ACT 1 MD 2
        a, b, c = self.coefficients(self.bn1_mu, self.bn1_sigma, self.bn1_gamma, self.bn1_beta, self.bn1_eps, 'CBA')

        bn_act_1_start_time = time.time()
        bn_act_1 = self.bn_act(conv1, a, b, c)
        bn_act_1_end_time = time.time()
        bn_act_1_time = bn_act_1_end_time - bn_act_1_start_time
        print(f'BatchNormalization-Activation time: {bn_act_1_time} s')

        # Conv2 MD 2
        conv2_convolution_start_time = time.time()
        conv2 = self.convolution(bn_act_1, self.conv2_weight, self.conv2_bias, self.conv2_stride)
        conv2_convolution_end_time = time.time()
        conv2_convolution_time = conv2_convolution_end_time - conv2_convolution_start_time
        print(f'Conv2 Convolution time: {conv2_convolution_time:.4f} s')

        # BN + ACT 2 MD 2
        a, b, c = self.coefficients(self.bn2_mu, self.bn2_sigma, self.bn2_gamma, self.bn2_beta, self.bn2_eps, "CBA")
        bn_act_2_start_time = time.time()
        bn_act_2 = self.bn_act(conv2, a, b, c)
        bn_act_2_end_time = time.time()
        bn_act_2_time = bn_act_2_end_time - bn_act_2_start_time
        print(f'BatchNormalization-Activation time: {bn_act_2_time} s')

        # Pack vector MD 1
        pack_start_time = time.time()
        x_pac = self.pack_vectors(bn_act_2, self.parms['n'])
        pack_end_time = time.time()
        pack_time = pack_end_time - pack_start_time
        print(f'Pack vector time: {pack_time} s')

        # FC 1 MD 1
        fc1_start_time = time.time()
        fc1 = self.fully_connected(x_pac, self.fc1_weight, self.fc1_bias)
        fc1_end_time = time.time()
        fc1_time = fc1_end_time - fc1_start_time
        print(f'FC1 time: {fc1_time} s')

        end = time.time()
        print(f'Run time/per inference:{end - start}s')
        return fc1

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

def Encryption_Parameters():
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 16384
    parms.set_poly_modulus_degree(poly_modulus_degree)
    bit_scale = 30
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, [60, bit_scale, bit_scale, bit_scale,
                              bit_scale, bit_scale, bit_scale,bit_scale, bit_scale, bit_scale, 60]))
    context = SEALContext(parms)
    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()

    parms = {
        'context': context,
        'encryptor': Encryptor(context, public_key),
        'evaluator': Evaluator(context),
        'decryptor': Decryptor(context, secret_key),
        'encoder': CKKSEncoder(context),
        'scale': 2.0 ** bit_scale,
        'galois_keys': keygen.create_galois_keys(),
        'relin_keys': keygen.create_relin_keys(),
        'slots': CKKSEncoder(context).slot_count()
    }

    context_data = context.key_context_data()
    print(f' | Encrption parameters: \n'
          f' | scheme: CKKS\n'
          f' | poly_modulus_degree: {context_data.parms().poly_modulus_degree()}')
    print(f' | coeff_modulus size: {context_data.total_coeff_modulus_bit_count()}')

    return parms

def encrypted_model(parms):

    PATH = f'Parameters/mnist_TenSEAL_ConvNet_100.pth'
    load_model = torch.load(PATH)
    model = TenSEAL_ConvNet()
    print(
        f'{model}'
    )
    model.load_state_dict(load_model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    '''
    test_data = datasets.MNIST('data', train=False, download=False, transform=transform)
    # test_data = datasets.CIFAR10('data', train=False, download=False, transform=transform)
    data = torch.utils.data.RandomSampler(test_data, replacement=True, num_samples=10)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, sampler=data)

    parms['m'] = 28 # mnist 28x28
    enc_model = EncBN_CNN(model, parms)
    print(
        f'EncNet\n {enc_model}'
    )
    criterion = torch.nn.CrossEntropyLoss()
    enc_test(enc_model, test_loader, criterion, parms)

# Encrypted data test
def enc_test(enc_model, test_loader, criterion, parms):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    start_time = time.time()
    inference = 0
    dict_time = {}
    pbar = tqdm(test_loader)
    # enc_model.eval()
    for j, (data, target) in enumerate(pbar):
        # Encoding and encryption
        x = []
        for i in range(data.shape[1]):
            x.append(parms['encryptor'].encrypt(
                parms['encoder'].encode(data[0][i].flatten().tolist(), parms['scale'])))

        # Encryption evaluation
        enc_output = enc_model(x, data)

        # Decryption of result
        output = parms['encoder'].decode(parms['decryptor'].decrypt(enc_output))
        # cut and obtain valid values
        output = torch.tensor(output[:10]).view(1, -1)
        # compute loss
        loss = criterion(output, target)
        test_loss += loss.item()

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        print(f'Target:{target} Pred:{pred} Loss:{loss}')

        # compare prediction to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1
        print(f'Temporary Accuracy:  {100 * np.sum(class_correct) / np.sum(class_total)}% ({int(np.sum(class_correct))}/{int(np.sum(class_total))})')
    end_time = time.time()

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader)
    print(f'Test Loss:{test_loss:.6f}\n')
    '''
    for label in range(10):
        print(
            f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}%'
            f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
        )
    '''
    print(
        f'\bTest Accuracy (Overall): {100 * np.sum(class_correct) / np.sum(class_total)}%'
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
        f'Run time: {end_time - start_time}s'
    )

def raw():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    '''

    train_dataset = torchvision.datasets.MNIST('data', train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.MNIST('data', train=False, transform=transform, download=False)
    # train_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=transform, download=False)
    # test_dataset = torchvision.datasets.CIFAR10('data', train=False, transform=transform, download=False)
    batch_size = 100
    learning_rate = 0.001

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # data = torch.utils.data.RandomSampler(test_dataset, replacement=True, num_samples=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=4)  # ,  sampler=data)

    # PATH = f'Parameters/mnist_MNIST_BN_CNN_100.pth'
    # load_model = torch.load(PATH)
    net = TenSEAL_ConvNet()
    print(f'{net}')
    # net.load_state_dict(load_model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    '''
    cpu_num = cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUN_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    '''

    net = train(net, train_loader, optimizer, criterion)
    test(net, test_loader, criterion)

if __name__ == '__main__':
    # raw()
    parms = Encryption_Parameters()
    encrypted_model(parms)


