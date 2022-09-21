from seal import *
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
# import tenseal as ts
import time
from tqdm import tqdm
import math
import pickle
from Prototype import TenSEAL_ConvNet

# model
class LeNet(torch.nn.Module):
    def __init__(self, input_channel=3, output=10):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        # self.fc1 = torch.nn.Linear(256, 120)
        self.fc1 = torch.nn.Linear(400, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, output)

    def forward(self, x):
        conv1 = F.max_pool2d(F.relu(self.conv1(x)), 2)
        conv2 = F.max_pool2d(F.relu(self.conv2(conv1)), 2)
        conv2 = conv2.view(conv2.size()[0], -1)
        fc1 = F.relu(self.fc1(conv2))
        fc2 = F.relu(self.fc2(fc1))
        fc3 = self.fc3(fc2)
        return fc3

class EncLeNet:
    def __init__(self, torch_nn, parms):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
             torch_nn.conv1.out_channels, torch_nn.conv1.in_channels, torch_nn.conv1.kernel_size[0] * torch_nn.conv1.kernel_size[1]
        ) # out, in, fxf
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        self.conv1_stride = torch_nn.conv1.stride[0]

        self.conv2_weight = torch_nn.conv2.weight.data.view(
            torch_nn.conv2.out_channels, torch_nn.conv2.in_channels, torch_nn.conv2.kernel_size[0] * torch_nn.conv2.kernel_size[1]
        ) # out, in, fxf
        self.conv2_bias = torch_nn.conv2.bias.data.tolist()
        self.conv2_stride = torch_nn.conv2.stride[0]

        self.fc1_weight = torch_nn.fc1.weight.data.T.tolist()
        self.fc1_weight = self.weight_diag(self.fc1_weight)
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.data.T.tolist()
        self.fc2_weight = self.weight_diag(self.fc2_weight)
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

        self.fc3_weight = torch_nn.fc3.weight.data.T.tolist()
        self.fc3_weight = self.weight_diag(self.fc3_weight)
        self.fc3_bias = torch_nn.fc3.bias.data.tolist()

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
                if i == f-1 & j == f-1:
                    continue
                a = start + j
                if j == f-1:
                    a = (i+1) * m - 1
                    idx.append(a+1)
                else:
                    idx.append(a+1)
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
        if s == 1:
            y = x
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
        self.parms['n'] = int((self.parms['m'] + 2 * p - f)/s + 1)

        # search rotation index
        idx = self.count_rotation(self.parms['m'], f)

        for outer in range(out_channel):
            Z = []
            for inner in range(in_channel):
                Y = []
                temp = input[inner]
                # Rotation and Accumulation
                for i in range(f*f):
                    #
                    # HE multiplication
                    # encode weight
                    #weight = self.parms['encoder'].encode([conv_weight[outer][inner][i].tolist() for a in range(self.parms['m']*self.parms['m'])], self.parms['scale'])
                    weight = self.parms['encoder'].encode(conv_weight[outer][inner][i].item(), self.parms['scale'])
                    self.parms['evaluator'].mod_switch_to_inplace(weight, input[inner].parms_id())
                    # Z = A * f_i, ciphertext * plaintext
                    Y.append(self.parms['evaluator'].multiply_plain(temp, weight))
                    if i != f*f -1:
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
            # bias = self.parms['encoder'].encode([conv_bias[outer] for a in range(self.parms['n']*self.parms['n'])], self.parms['scale'])
            bias = self.parms['encoder'].encode([conv_bias[outer] for a in range(self.parms['n'] * self.parms['n'])], self.parms['scale'])
            self.parms['evaluator'].mod_switch_to_inplace(bias, Z.parms_id())
            Z.scale(2**int(math.log2(bias.scale())))
            self.parms['evaluator'].add_plain_inplace(Z, bias)
            out_channels.append(Z)

        # set output of layer as input of next layer
        self.parms['m'] = self.parms['n']
        return out_channels

    def acitvation(self, x, name):
        channels = len(x)
        for i in range(channels):
            if name =='square':
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
                idx.append(i*s*m + j*s)
        for j in range(1, n*n):
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
            # add first channel
            Y.append(x[i])
            for j in range(1, f):
                # HE rotation and addition
                Y.append(self.parms['evaluator'].rotate_vector(x[i], j*self.parms['m'], self.parms['galois_keys']))
            y = self.parms['evaluator'].add_many(Y)
            # rotate one by one of raw
            Y = []
            Y.append(y)
            for j in range(0, f-1):
                # HE rotation and addtition
                # self.parms['evaluator'].rotate_vector_inplace(y, j+1, self.parms['galois_keys'])
                Y.append(self.parms['evaluator'].rotate_vector(y, j + 1, self.parms['galois_keys']))

            y = self.parms['evaluator'].add_many(Y)


            # idx: valuable idx; rot_idx: rotation index
            idx, rot_idx = self.start_index(self.parms['m'], self.parms['n'], s)
            # generate all-zeroes mask
            mask = [0 for h in range(self.parms['slots'])]
            mask[0] = 1/(f*f)
            mask = self.parms['encoder'].encode(mask, self.parms['scale'])
            self.parms['evaluator'].mod_switch_to_inplace(mask, y.parms_id())
            Z =[]
            Z.append(self.parms['evaluator'].multiply_plain(y, mask))
            for k in range(self.parms['n']*self.parms['n']-1):
                #HE element-wise multiplication and addition
                mask = [0 for h in range(self.parms['slots'])]
                mask[idx[k+1]] = 1 / (f * f)
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
        weight = np.pad(weight, pad_width=((0,0), (0, weight.shape[0]-weight.shape[1])))
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


    def forward(self, enc_x): # try to reduce paramteres for functions
        self.parms['m'] = 28
        # self.parms['m'] = 32
        # MD 2 convolution
        # self.conv1_weight = self.weight_change(self.conv1_weight, self.parms['m'])
        # enc_x = self.pack_vactors(enc_x, self.parms['m'])
        conv1_convolution_start_time = time.time()
        conv1 = self.convolution(enc_x, self.conv1_weight, self.conv1_bias, self.conv1_stride) # try to built-in m
        # conv1 = self.conv(enc_x, self.conv1_weight, self.conv1_bias, self.conv1_stride)  # try to built-in m
        conv1_convolution_end_time = time.time()
        conv1_convolution_time = conv1_convolution_end_time - conv1_convolution_start_time
        print(f'\nConv1 Convolution time: {conv1_convolution_time:.4f} s')

        # MD 1 activation
        conv1_act_start_time = time.time()
        conv1_act = self.acitvation(conv1, 'square')
        conv1_act_end_time = time.time()
        conv1_act_time = conv1_act_end_time - conv1_act_start_time
        print(f'\nConv1 Activation time: {conv1_act_time} s')

        # MD 1 average-pooling
        conv1_avg_start_time = time.time()
        conv1_avg = self.average_pooling(conv1_act, 2, 2) #  try to built-in parameters
        conv1_avg_end_time = time.time()
        conv1_avg_time = conv1_avg_end_time - conv1_avg_start_time
        print(f'\nConv1 Avg-pool time: {conv1_avg_time:.2f} s')

        # MD 2 convolution
        conv2_convolution_start_time = time.time()
        conv2 = self.convolution(conv1_avg, self.conv2_weight, self.conv2_bias, self.conv2_stride)
        conv2_convolution_end_time =time.time()
        conv2_convolution_time = conv2_convolution_end_time - conv2_convolution_start_time
        print(f'\nConv2 Convolution time: {conv2_convolution_time:.4f} s')

        # MD 1 activation
        conv2_act_start_time = time.time()
        conv2_act = self.acitvation(conv2, 'square')
        conv2_act_end_time = time.time()
        conv2_act_time = conv2_act_end_time - conv2_act_start_time
        print(f'\nConv2 Activation time: {conv2_act_time} s')

        # MD 1 average-pooling
        conv2_avg_start_time = time.time()
        conv2_avg = self.average_pooling(conv2_act, 2, 2)
        conv2_avg_end_time = time.time()
        conv2_avg_time = conv2_avg_end_time - conv2_avg_start_time
        print(f'\nConv2 Avg-pool time: {conv2_avg_time:.2f} s')

        # Pack vector
        pack_start_time = time.time()
        x_pac = self.pack_vectors(conv2_avg, self.parms['n'])
        pack_end_time = time.time()
        pack_time = pack_end_time - pack_start_time
        print(f'Pack vector time: {pack_time} s')

        # MD 1 fc
        fc1_start_time = time.time()
        fc1 = self.fully_connected(x_pac, self.fc1_weight, self.fc1_bias)
        fc1_end_time = time.time()
        fc1_time = fc1_end_time - fc1_start_time
        print(f'FC1 time: {fc1_time} s')

        # MD 1 fc
        fc2_start_time = time.time()
        fc2 = self.fully_connected(fc1, self.fc2_weight, self.fc2_bias)
        fc2_end_time = time.time()
        fc2_time = fc2_end_time - fc2_start_time
        print(f'FC2 time: {fc2_time} s')

        # MD 1 fc
        fc3_star_time = time.time()
        fc3 = self.fully_connected(fc2, self.fc3_weight, self.fc3_bias)
        fc3_end_time = time.time()
        fc3_time = fc3_end_time - fc3_star_time
        print(f'FC3 time: {fc3_time} s')
        
        runtime = {}
        runtime['conv1_convolution_time'] = conv1_convolution_time
        runtime['conv1_act_time'] = conv1_act_time
        runtime['conv1_avg_time'] = conv1_avg_time

        runtime['conv2_convolution_time'] = conv2_convolution_time
        runtime['conv2_act_time'] = conv2_act_time
        runtime['conv2_avg_time'] = conv2_avg_time
        
        runtime['pack_time'] = pack_time
        
        runtime['fc1_time'] = fc1_time
        runtime['fc2_time'] = fc2_time
        runtime['fc3_time'] = fc3_time
        
        runtime['total'] = conv1_convolution_time + conv1_act_time + conv1_avg_time +\
                        conv2_convolution_time + conv2_act_time + conv2_avg_time +\
                        pack_time + fc1_time + fc2_time + fc3_time
       
        return fc3, runtime

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

def Encryption_Parameters():
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 16384
    parms.set_poly_modulus_degree(poly_modulus_degree)
    bit_scale = 30
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, [50, bit_scale, bit_scale, bit_scale, bit_scale,
                              bit_scale, bit_scale, bit_scale, bit_scale,
                              bit_scale, bit_scale, bit_scale, 50]))
    context = SEALContext(parms)
    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()

    parms = {
        'context' : context,
        'encryptor': Encryptor(context, public_key),
        'evaluator': Evaluator(context),
        'decryptor': Decryptor(context, secret_key),
        'encoder': CKKSEncoder(context),
        'scale': 2.0 ** bit_scale,
        'galois_keys': keygen.create_galois_keys(),
        'relin_keys' : keygen.create_relin_keys(),
        'slots': CKKSEncoder(context).slot_count()
    }
    return parms

def encrypted_model(parms):
    # PATH = f'Parameters/Alter_mnist_LeNet_100.pth'
    # PATH = f'Parameters/Alter_cifar10_LeNet_500.pth'
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

    # parms['m'] = 32 # mnist 28x28
    enc_model = EncLeNet(model, parms)
    print(
        f'EncLeNet\n {enc_model}'
    )
    criterion = torch.nn.CrossEntropyLoss()
    enc_test(enc_model, test_loader, criterion, parms)


# Encrypted data test
def enc_test(enc_model, test_loader, criterion, parms):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # start_time = time.time()
    inference = 0
    dict_time = {}
    pbar = tqdm(test_loader)
    # enc_model.eval()
    for j , (data, target) in enumerate(pbar):
        # Encoding and encryption
        x = []
        for i in range(data.shape[1]):
            x.append(parms['encryptor'].encrypt(
                parms['encoder'].encode(data[0][i].flatten().tolist(), parms['scale'])))

        # Encryption evaluation
        enc_output, inf_time = enc_model(x)
        inference += inf_time['total']
        dict_time.update({i:inf_time})
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
    # end_time = time.time()
    with open('Parameters/total_'+ str(j+1)+'.pkl', 'wb') as f:
        pickle.dump(dict_time, f, pickle.HIGHEST_PROTOCOL)

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
        f'\bTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}%'
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    )
    print(
        f'Inference Time: {inference:.6f}s ; {inference / int(j+1):.6f} s/per'
    )


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parms = Encryption_Parameters()
    encrypted_model(parms)























