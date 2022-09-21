import pickle
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from seal import *
from tqdm import tqdm
import time
from multiprocessing import cpu_count
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import dill

"""
def weight_change(weight, m):
    '''
    f = int(math.sqrt(weight[0].__len__()))
    input_channel = int(weight.__len__())
    weight = list(map(list, zip(*weight)))
    x = [[] for i in range(f*f)]
    for i in range(f*f):
        for j in range(input_channel):
            x[i].extend([weight[i][j] for y in range(m*m)])
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
def get_diagonal(position, matrix):
    n = matrix.shape[0]
    diagonal = np.zeros(n)

    k = 0
    i = 0
    j = position
    while i < n-position and j < n:
        diagonal[k] = matrix[i][j]
        i += 1
        j += 1
        k += 1

    i = n - position
    j = 0
    while i < n and j < position:
        diagonal[k] = matrix[i][j]
        i += 1
        j += 1
        k += 1

    return diagonal
def get_all_diagonals(matrix):
    matrix_diagonals = []
    for i in range(matrix.shape[0]):
        matrix_diagonals.append(get_diagonal(i, matrix))

    return np.array(matrix_diagonals)
def get_u_sigma(shape):
    u_sigma_ = np.zeros(shape)
    indices_diagonal = np.diag_indices(shape[0])
    u_sigma_[indices_diagonal] = 1.

    for i in range(shape[0]-1):
        u_sigma_ = np.pad(u_sigma_, (0, shape[0]), 'constant')
        temp = np.zeros(shape)
        j = np.arange(0, shape[0])
        temp[j, j-(shape[0]-1-i)] = 1.
        temp = np.pad(temp, ((i+1)*shape[0], 0), 'constant')
        u_sigma_ += temp

    return u_sigma_
def get_u_tau(shape):
    u_tau_ = np.zeros((shape[0], shape[0]**2))
    index = np.arange(shape[0])
    for i in range(shape[0], 0, -1):
        idx = np.concatenate([index[i:], index[:i]], axis=0)
        row = np.zeros(shape)
        for j in range(shape[0]):
            temp = np.zeros(shape)
            temp[idx[j], idx[j]] = 1.
            if j == 0:
                row += temp
            else:
                row = np.concatenate([row, temp], axis=1)

        if i == shape[0]:
            u_tau_ += row
        else:
            u_tau_ = np.concatenate([u_tau_, row], axis=0)

    return u_tau_
def get_v_k(shape):
    v_k_ = []
    index = np.arange(0, shape[0])
    for j in range(1, shape[0]):
        temp = np.zeros(shape)
        temp[index, index-(shape[0]-j)] = 1.
        mat = temp
        for i in range(shape[0]-1):
            mat = np.pad(mat, (0, shape[0]), 'constant')
            temp2 = np.pad(temp, ((i+1)*shape[0], 0), 'constant')
            mat += temp2

        v_k_.append(mat)

    return v_k_
def get_w_k(shape):
    w_k_ = []
    index = np.arange(shape[0]**2)
    for i in range(shape[0]-1):
        temp = np.zeros((shape[0]**2, shape[1]**2))
        temp[index-(i+1)*shape[0], index] = 1.
        w_k_.append(temp)

    return w_k_
def linear_transform_plain(cipher_matrix, plain_diags, galois_keys, evaluator):
    cipher_rot = evaluator.rotate_vector(cipher_matrix, -len(plain_diags), galois_keys)
    cipher_temp = evaluator.add(cipher_matrix, cipher_rot)
    cipher_results = []
    temp = evaluator.multiply_plain(cipher_temp, plain_diags[0])
    cipher_results.append(temp)

    i = 1
    while i < len(plain_diags):
        temp_rot = evaluator.rotate_vector(cipher_temp, i, galois_keys)
        temp = evaluator.multiply_plain(temp_rot, plain_diags[i])
        cipher_results.append(temp)
        i += 1

    cipher_prime = evaluator.add_many(cipher_results)

    return cipher_prime

if __name__ == '__main__':
    x = np.arange(1, 10).reshape(3,3)

    y = np.array([1, 2, 3])


    u_sigma = get_u_sigma(x.shape)
    u_tau = get_u_tau(x.shape)
    v_k = get_v_k(x.shape)
    w_k = get_w_k(x.shape)

    print(x)
    print('U_sigma')
    print(u_sigma)
    print('U_tau')
    print(u_tau)
    print('V_k')
    print(v_k)
    print('W_k')
    print(w_k)

    parms = Encryption_Parameters()

    plain_u_sigma_diagonals = []
    # plain_u_tau_diagonals = []
    n = 3

    u_sigma_diagonals = get_all_diagonals(u_sigma)
    u_sigma_diagonals += 0.000001

    print('U_sigma_diagonals')
    print(u_sigma_diagonals)

    for i in range(n ** 2):
        plain_u_sigma_diagonals.append(parms['encoder'].encode(u_sigma_diagonals[i], parms['scale']))
        # plain_u_tau_diagonals.append(parms['encoder'].encode(u_tau_diagonals[i], parms['scale']))

    plain_matrix1 = parms['encoder'].encode(x.flatten(), parms['scale'])
    cipher_matrix1 = parms['encryptor'].encrypt(plain_matrix1)

    ltp = linear_transform_plain(cipher_matrix1, plain_u_sigma_diagonals, parms['galois_keys'], parms['evaluator'])
    print(ltp)
"""

def compare_loss():
    net = CompareNet()
    print(f'{net}')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_data = datasets.MNIST('data', train=False, download=False, transform=transform)
    data = torch.utils.data.RandomSampler(test_data, replacement=True, num_samples=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, sampler=data)
    criterion = torch.nn.CrossEntropyLoss()
    pbar = tqdm(test_loader)
    for j, (x, y) in enumerate(pbar):
        avg, enc_avg = net(x)
        enc_avg = torch.tensor(enc_avg).view(1, -1)

        avg = avg.flatten().tolist()
        enc_avg = enc_avg.flatten().tolist()

        p_avg = np.array(avg) / np.sum(avg)
        p_enc_avg = np.array(enc_avg) / np.sum(enc_avg)

        # KL = scipy.stats.entropy(avg, enc_avg)
        # avg_loss = criterion(avg.view(1, -1), y)
        # enc_avg_loss = criterion(enc_avg, y)

        print(f'KL')

def rot_conv(m, f):
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

def rot_mask(m, n, f, s):
    idx = []
    rot_idx = []
    for i in range(n):
        for j in range(n):
            idx.append(i * s * m + j * s)
    for j in range(1, n * n):
        rot_idx.append(idx[j] - j)
    return idx, rot_idx

class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)# 64
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        conv1 = self.conv1(x)
        act = conv1 * conv1
        act = act.view(-1, 256)
        fc1 = self.fc1(act)
        fc1 = fc1 * fc1
        fc2 = self.fc2(fc1)
        return fc2


if __name__ == '__main__':

    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 16384
    parms.set_poly_modulus_degree(poly_modulus_degree)
    bit_scale = 30
    parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, bit_scale, bit_scale, 60]))
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

    arr = [1, 2, 3, 4, 5]
    arr_encode = parms['encoder'].encode(arr, parms['scale'])
    arr_enc = parms['encryptor'].encrypt(arr_encode)
    c5 = parms['evaluator'].rotate_vector(arr_enc, 1, parms['galois_keys'])
    c6 = parms['evaluator'].rotate_vector(arr_enc, -1, parms['galois_keys'])

    arr_dec = parms['encoder'].decode(parms['decryptor'].decrypt(c5))
    arr_dec1 = parms['encoder'].decode(parms['decryptor'].decrypt(c6))


    print(f'c5: {arr_dec[0:5]}')
    print(f'c6: {arr_dec1[0:5]}')

    #print(f'c1: {c1_dec[0]}')
    #print(f'c2: {c2_dec[0]}')
    #print(f'c3: {c3_dec[0]}')
    #print(f'c4: {c4_dec[0]}')










    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    context_data = context.key_context_data()
    print(f' | Encrption parameters: \n'
          f' | scheme: CKKS\n'
          f' | poly_modulus_degree: {context_data.parms().poly_modulus_degree()}')
    print(f' | coeff_modulus size: {context_data.total_coeff_modulus_bit_count()}')
    coeff_modulus = context_data.parms().coeff_modulus()
    coeff_modulus_size = coeff_modulus.__len__()

    
    test_data = datasets.MNIST('data', train=False, download=False, transform=transform)
    data = torch.utils.data.RandomSampler(test_data, replacement=True, num_samples=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, sampler=data)

    
    # MNIST Pixel-based Encryption
    s_time = time.time()
    for (data,target) in test_loader:
        data = data.flatten().tolist()
        for i in range(len(data)):
            data[i] = parms['encryptor'].encrypt(
                parms['encoder'].encode(data[i], parms['scale']))
    e_time = time.time()
    print(f'Pixel-based Encryption time {e_time - s_time}s')

    # MNIST Pixel-based Decryption
    de_s_time = time.time()
    for i in range(len(data)):
        data[i] = parms['encoder'].decode(parms['decryptor'].decrypt(data[i]))
    de_e_time = time.time()
    print(f'Pixel-based Decryption time {de_e_time - de_s_time}s')

    # MNIST Channel-based Encryption
    c_s_time = time.time()
    for (data,target) in test_loader:
        data = data.flatten().tolist()
        data = parms['encryptor'].encrypt(
                parms['encoder'].encode(data, parms['scale']))
    c_e_time = time.time()
    print(f'Channel-based Encryption time {c_e_time - c_s_time}s')

    # MNIST Channel-based Decrytpion
    de_c_s_time = time.time()
    data = parms['encoder'].decode(parms['decryptor'].decrypt(data))
    de_c_e_time = time.time()
    print(f'Channel-based Decryption time {de_c_e_time - de_c_s_time}s')

    Encryption_time = {}
    Encryption_time['MNIST_Pixel-based_Encryption'] = (e_time - s_time)
    Encryption_time['MNIST_Channel-based_Encryption'] = (c_e_time - c_s_time)

    Decryption_time = {}
    Decryption_time['MNIST_Pixel-based_Decryption'] = (de_e_time - de_s_time)
    Decryption_time['MNIST_Channel-based_Decryption'] = (de_c_e_time - de_c_s_time)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_data = datasets.CIFAR10('data', train=False, download=False, transform=transform)
    data = torch.utils.data.RandomSampler(test_data, replacement=True, num_samples=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, sampler=data)

    # CIFAR-10 Pixel-based Encryption
    s_time = time.time()
    for (data, target) in test_loader:
        data = data.flatten().tolist()
        for i in range(len(data)):
            data[i] = parms['encryptor'].encrypt(
                parms['encoder'].encode(data[i], parms['scale']))
    e_time = time.time()
    print(f'CIFAR-10 Pixel-based Encryption time {e_time - s_time}s')

    # CIFAR-10 Pixel-based Decryption
    de_p_s_time = time.time()
    for i in range(len(data)):
        data[i] = parms['encoder'].decode(parms['decryptor'].decrypt(data[i]))
    de_p_e_time = time.time()
    print(f'CIFAR-10 Pixel-based Decryption time {de_p_e_time - de_p_s_time}s')

    # CIFAR-10 Channel-based Encryption
    c_s_time = time.time()
    for (data, target) in test_loader:
        # data = data.flatten().tolist()
        x = []
        for i in range(data.shape[1]):
            x.append(parms['encryptor'].encrypt(
                parms['encoder'].encode(data[0][i].flatten().tolist(), parms['scale'])))
    c_e_time = time.time()
    print(f'CIFAR-10 Channel-based Encryption time {c_e_time - c_s_time}s')

    # CIFAR-10 Channel-based Decryption
    de_c_c_s_time = time.time()
    for i in range(len(x)):
        x[i] = parms['encoder'].decode(parms['decryptor'].decrypt(x[i]))
    de_c_c_e_time = time.time()
    print(f'CIFAR-10 Channel-based Decryption time {de_c_c_e_time - de_c_c_s_time}s')

    Encryption_time['CIFAR10_Pixel-based_Encryption'] = (e_time - s_time)
    Encryption_time['CIFAR10_Channel-based_Encryption'] = (c_e_time - c_s_time)

    Decryption_time['CIFAR10_Pixel-based_Decryption'] = (de_p_e_time - de_p_s_time)
    Decryption_time['CIFAR10_Channel-based_Decryption'] = (de_c_c_e_time - de_c_c_s_time)
    np.save('Encryption_time.npy', Encryption_time)
    np.save('Decryption_time.npy', Decryption_time)
    """