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
from Models.LeNet import LeNet
from seal import *
import math
import scipy.stats

# model
class Avgnn(torch.nn.Module):
    def __init__(self, input_channel=1, output=10):
        super(Avgnn, self).__init__()
        self.fc = torch.nn.Linear(196,10)

    def forward(self, x):
        avg = F.avg_pool2d(x, 2)
        x_pac = avg.view(avg.size()[0], -1)
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
    net.train()
    device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")
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


class EncAcnn:
    def __init__(self, torch_nn, parms):

        self.fc1_weight = torch_nn.fc.weight.data.T.tolist()
        self.fc1_weight = self.weight_diag(self.fc1_weight)
        self.fc1_bias = torch_nn.fc.bias.data.tolist()

        # parameter dictionary
        self.parms = parms

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

    def mask_computation(self, x, n, f, s):
        """
        :param x: input  HE encrypted vector
        :param n: side of output
        :param f: side of filter
        :param g: gap
        :param N: polynomial modulus degree
        :return: well-structure vector
        """
        y = x
        if s == 1:
            z = [0 for h in range(parms['slots'])]
            z[:n] = [1 for h in range(n)]
            z = self.parms['encoder'].encode(z, self.parms['scale'])
            parms['evaluator'].mod_switch_to_inplace(z, x.parms_id())
            # HE multiplication
            Z = []
            Z.append(self.parms['evaluator'].multiply_plain(y, z))

            for i in range(n - 1):
                h = [0 for x in range(self.parms['slots'])]
                h[(i + 1) * n:(i + 2) * n] = [1 for u in range(n)]
                h = self.parms['encoder'].encode(h, self.parms['scale'])
                parms['evaluator'].mod_switch_to_inplace(h, x.parms_id())

                # HE rotation
                self.parms['evaluator'].rotate_vector_inplace(y, self.parms['gap'], self.parms['galois_keys'])
                # HE multiplication and addition
                Z.append(self.parms['evaluator'].multiply_plain(y, h))
            Z = self.parms['evaluator'].add_many(Z)
        else:
            # idx: valuable idx; rot_idx: rotation index
            idx, rot_idx = self.start_index(self.parms['m'], self.parms['n'], s)
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

    def weight_change(self, weight, m):
        '''
        change noraml weight matrix to effective weight matrix for conv, in order to reduce times of rotation
        :param weight: weight parameter input_channle X kernel_size(fxf)
        :param m: side of input feature map
        :return: changed weight matrix for conv, out X fxf X mxmxin
        '''
        '''
        f = int(math.sqrt(weight[0].__len__()))
        input_channel = int(weight.__len__())
        weight = list(map(list, zip(*weight)))
        x = [[] for i in range(f * f)]
        for i in range(f * f):
            for j in range(input_channel):
                x[i].extend([weight[i][j] for y in range(m * m)])
        return x
        '''
        f = int(math.sqrt(weight.shape[2]))
        input_channel = weight.shape[1]
        output_channel = weight.shape[0]
        weight = weight.permute(0, 2, 1)
        weight_out = []
        X = [[[] for i in range(f * f)] for j in range(output_channel)]
        for out in range(output_channel):
            for i in range(f * f):
                for j in range(input_channel):
                    X[out][i].extend([weight[out][i][j] for y in range(m * m)])
        return torch.Tensor(X)

    def convolution(self, input, conv_weight, conv_bias, s):
        """
        :param input: vector include encrypted data: should be squeezed for simplicity
        :param conv_weight: convolution layer weight
        :param conv_bias: convolution layer bias
        :param m: side of input
        :return: after convolution computation
        """
        out_channels = []
        in_channel = conv_weight.shape[1]
        out_channel = conv_weight.shape[0]
        f = int(math.sqrt(conv_weight.shape[2]))
        p = 0
        self.parms['n'] = int((self.parms['m'] + 2 * p - f) / s + 1)

        # search rotation index
        idx = self.count_rotation(self.parms['m'], f)

        for outer in range(out_channel):
            Z = []
            for inner in range(in_channel):
                Y = []
                temp = input[inner]
                # Rotation and Accumulation
                for i in range(f * f):
                    #
                    # HE multiplication
                    # encode weight

                    weight = self.parms['encoder'].encode(
                        [conv_weight[outer][inner][i].tolist() for a in range(self.parms['m'] * self.parms['m'])],
                        self.parms['scale'])
                    self.parms['evaluator'].mod_switch_to_inplace(weight, input[inner].parms_id())
                    # Z = A * f_i, ciphertext * plaintext
                    Y.append(self.parms['evaluator'].multiply_plain(temp, weight))
                    if i != f * f - 1:
                        # HE rotation
                        temp = self.parms['evaluator'].rotate_vector(input[inner], idx[i], self.parms['galois_keys'])

                # Mask Computation
                # HE add all in list
                Y = self.parms['evaluator'].add_many(Y)
                self.parms['evaluator'].rescale_to_next_inplace(Y)
                self.parms['gap'] = self.parms['m'] - self.parms['n']
                Y = self.mask_computation(Y, self.parms['n'], f, s)
                Z.append(Y)

            # Add Bias
            Z = self.parms['evaluator'].add_many(Z)
            self.parms['evaluator'].rescale_to_next_inplace(Z)

            # HE addition
            # encode bias
            bias = self.parms['encoder'].encode([conv_bias[outer] for a in range(self.parms['n'] * self.parms['n'])],
                                                self.parms['scale'])
            self.parms['evaluator'].mod_switch_to_inplace(bias, Z.parms_id())
            Z.scale(2 ** int(math.log2(bias.scale())))
            self.parms['evaluator'].add_plain_inplace(Z, bias)
            out_channels.append(Z)

        # set output of layer as input of next layer
        self.parms['m'] = self.parms['n']
        return out_channels

    def acitvation(self, x, name):
        channels = len(x)
        for i in range(channels):
            if name == 'square':
                # HE square
                self.parms['evaluator'].square_inplace(x[i])
                self.parms['evaluator'].relinearize_inplace(x[i], self.parms['relin_keys'])
                self.parms['evaluator'].rescale_to_next_inplace(x[i])
        return x

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
        for i in range(out_channel):
            # y = x[i]
            # rotate by raw
            Y = []
            Y.append(x[i])
            for j in range(1, f):
                # HE rotation and addition
                Y.append(self.parms['evaluator'].rotate_vector(x[i], j * self.parms['m'], self.parms['galois_keys']))
            y = self.parms['evaluator'].add_many(Y)
            # rotate one by one of raw
            Y = []
            Y.append(y)
            for j in range(0, f - 1):
                # HE rotation and addtition
                self.parms['evaluator'].rotate_vector_inplace(y, j + 1, self.parms['galois_keys'])
                Y.append(y)

            y = self.parms['evaluator'].add_many(Y)

            # idx: valuable idx; rot_idx: rotation index
            idx, rot_idx = self.start_index(self.parms['m'], self.parms['n'], s)
            # generate all-zeroes mask
            mask = [0 for h in range(self.parms['slots'])]
            mask[0] = 1 / (f * f)
            mask = self.parms['encoder'].encode(mask, self.parms['scale'])
            self.parms['evaluator'].mod_switch_to_inplace(mask, y.parms_id())
            Z = []
            Z.append(self.parms['evaluator'].multiply_plain(y, mask))
            for k in range(self.parms['n'] * self.parms['n'] - 1):
                # HE element-wise multiplication and addition
                mask = [0 for h in range(self.parms['slots'])]
                mask[idx[k + 1]] = 1 / (f * f)
                mask = self.parms['encoder'].encode(mask, self.parms['scale'])
                self.parms['evaluator'].mod_switch_to_inplace(mask, y.parms_id())

                Z.append(
                    self.parms['evaluator'].rotate_vector(
                        self.parms['evaluator'].multiply_plain(y, mask),
                        rot_idx[k], self.parms['galois_keys']
                    )
                )
            Z = self.parms['evaluator'].add_many(Z)
            self.parms['evaluator'].rescale_to_next_inplace(Z)
            out_channels.append(Z)
        self.parms['m'] = self.parms['n']
        return out_channels

    def pack_vectors(self, x, n):
        channels = x.__len__()
        Y = []
        Y.append(x[0])
        for i in range(1, channels):
            # HE rotation
            Y.append(self.parms['evaluator'].rotate_vector(x[i], -(n * n * i), self.parms['galois_keys']))
        Y = self.parms['evaluator'].add_many(Y)
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
        for i in range(weight.__len__()):
            w = self.parms['encoder'].encode(weight[i], self.parms['scale'])
            self.parms['evaluator'].mod_switch_to_inplace(w, x.parms_id())
            Y.append(
                self.parms['evaluator'].multiply_plain(temp, w)
            )
            B = []
            if i != weight.__len__() - 1:
                B.append(self.parms['evaluator'].rotate_vector(x, i + 1, self.parms['galois_keys']))
                B.append(self.parms['evaluator'].rotate_vector(x, -weight.__len__() + i + 1, self.parms['galois_keys']))
                temp = self.parms['evaluator'].add_many(B)
        Y = self.parms['evaluator'].add_many(Y)
        self.parms['evaluator'].rescale_to_next_inplace(Y)

        B = self.parms['encoder'].encode(bias, self.parms['scale'])
        self.parms['evaluator'].mod_switch_to_inplace(B, Y.parms_id())
        Y.scale(2 ** int(math.log2(B.scale())))
        return self.parms['evaluator'].add_plain(Y, B)

    def forward(self, enc_x):  # try to reduce paramteres for functions
        self.parms['m'] = 28
        # self.parms['m'] = 32
        # MD  convolution 13

        # MD 1 average-pooling
        conv1_avg_start_time = time.time()
        conv1_avg = self.average_pooling(enc_x, 2, 2)  # try to built-in parameters
        conv1_avg_end_time = time.time()
        conv1_avg_time = conv1_avg_end_time - conv1_avg_start_time
        print(f'Avg-pool time: {conv1_avg_time:.2f} s')

        # Pack vector
        pack_start_time = time.time()
        x_pac = self.pack_vectors(conv1_avg, self.parms['n'])
        pack_end_time = time.time()
        pack_time = pack_end_time - pack_start_time
        print(f'Pack vector time: {pack_time} s')

        # FC1 md 1
        fc1_start_time = time.time()
        fc = self.fully_connected(x_pac, self.fc1_weight, self.fc1_bias)
        fc1_end_time = time.time()
        fc1_time = fc1_end_time - fc1_start_time
        print(f'FC1 time: {fc1_time} s')

        return fc

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

def encrypted_model(parms):
    # PATH = f'Parameters/Alter_mnist_LeNet_100.pth'
    # PATH = f'Parameters/Alter_cifar10_LeNet_500.pth'
    PATH = f'Parameters/mnist_Avgnn_100.pth'
    load_model = torch.load(PATH)
    model = Avgnn()
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
    # data = torch.utils.data.RandomSampler(test_data, replacement=True, num_samples=10000)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)#, sampler=data)

    # parms['m'] = 32 # mnist 28x28
    enc_model = EncAcnn(model, parms)
    print(
        f'EncAvgnn\n {enc_model}'
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
        enc_output = enc_model(x)

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

def Encryption_Parameters():
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    bit_scale = 30
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, [60, bit_scale, bit_scale, 60]))
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
    return parms

if __name__ == '__main__':
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
   

    train_dataset = torchvision.datasets.MNIST('data', train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.MNIST('data', train=False, transform=transform, download=False)
    # train_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=transform, download=False)
    # test_dataset = torchvision.datasets.CIFAR10('data', train=False, transform=transform, download=False)
    batch_size = 100
    learning_rate = 0.001
    epochs = 100
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # data = torch.utils.data.RandomSampler(test_dataset, replacement=True, num_samples=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4) #,  sampler=data)

    # PATH = f'Parameters/mnist_TenSEAL_ConvNet_100.pth'
    # load_model = torch.load(PATH)
    net = Avgnn()
    # net.load_state_dict(load_model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    net = train(net, train_loader, optimizer, criterion)
    test(net, test_loader, criterion)
    '''
    parms = Encryption_Parameters()
    encrypted_model(parms)