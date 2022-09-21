import torch
import torchvision
from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time
from tqdm import tqdm
from seal import *
import math
from itertools import product
from CHE.CHE import *
import pathos

# model


class ConvNet(torch.nn.Module):
    def __init__(self, output=10):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, kernel_size=5, padding=0, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(num_features=5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(5, 50, kernel_size=5, padding=0, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = torch.nn.Linear(800, 10)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        act1 = bn1.pow(2)
        conv2 = self.conv2(act1)
        bn2 = self.bn2(conv2)
        act2 = bn2.pow(2)
        flatten = act2.view(-1, 800)
        fc1 = self.fc1(flatten)
        return fc1


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
    PATH = f'Parameters/MNIST/CBA_{net.__class__.__name__}_{epochs}.pth'
    torch.save(net.state_dict(), PATH)
    return net


def Encryption_Parameters():
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 16384
    parms.set_poly_modulus_degree(poly_modulus_degree)
    bit_scale = 30
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, [39, bit_scale, bit_scale, bit_scale,
                              bit_scale, bit_scale, bit_scale, bit_scale, bit_scale, bit_scale, bit_scale, bit_scale, 39]))
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


def init(torch_nn, parms):
    model = {}
    model.update({
    "conv1_weight": weight_change(torch_nn.conv1.weight.data.view(
        torch_nn.conv1.out_channels, torch_nn.conv1.in_channels,
        torch_nn.conv1.kernel_size[0] * torch_nn.conv1.kernel_size[1]
    ), parms),
    "conv1_bias": bias_change(torch_nn.conv1.bias.data.tolist(),parms),
    "conv1_stride": torch_nn.conv1.stride[0],
    "conv2_weight": torch_nn.conv2.weight.data.view(
        torch_nn.conv2.out_channels, torch_nn.conv2.in_channels,
        torch_nn.conv2.kernel_size[0] * torch_nn.conv2.kernel_size[1]),
    "conv2_bias": bias_change(torch_nn.conv2.bias.data.tolist(),parms),
    "conv2_stride": torch_nn.conv2.stride[0],
    "bn1_mu": torch_nn.bn1.running_mean.data.numpy(),
    "bn1_sigma": torch_nn.bn1.running_var.data.numpy(),
    "bn1_gamma": torch_nn.bn1.weight.data.numpy(),
    "bn1_beta": torch_nn.bn1.bias.data.numpy(),
    "bn1_eps": torch_nn.bn1.eps,
    "bn2_mu": torch_nn.bn2.running_mean.data.numpy(),
    "bn2_sigma": torch_nn.bn2.running_var.data.numpy(),
    "bn2_gamma": torch_nn.bn2.weight.data.numpy(),
    "bn2_beta": torch_nn.bn2.bias.data.numpy(),
    "bn2_eps": torch_nn.bn2.eps,
    "fc1_weight": torch_nn.fc1.weight.data.T.tolist(),
    "fc1_bias": torch_nn.fc1.bias.data.tolist()
    })

    model["bn1_a"], model["bn1_b"], model["bn1_c"] = coefficients(model["bn1_mu"], model["bn1_sigma"], model["bn1_mu"], model["bn1_beta"], model["bn1_eps"], 'CBA')
    model["conv2_weight"] = weight_change(model["conv2_weight"]*model["bn1_a"].reshape(-1, 1), parms)
    model["bn2_a"], model["bn2_b"], model["bn2_c"] = coefficients(model["bn2_mu"], model["bn2_sigma"], model["bn2_mu"], model["bn2_beta"], model["bn2_eps"], 'CBA')
    # expand to fc number 800, each times 4x4=16
    model["fc1_weight"] = weight_diag(model["fc1_weight"] * np.tile(model["bn2_a"].reshape(-1, 1), (1, 16)).reshape(-1, 1))

    # valid vector
    # valid_vector = 0
    return model


def multiprocessing_model(enc_x, model, parms):
    start = time.time()
    # input size mxm
    parms["m"] = 28
    parms["valid_vector"] = [i for i in range(parms["m"] * parms["m"])]

    # Conv1 MD 2
    conv1_convolution_start_time = time.time()
    conv1 = convolution(enc_x, model["conv1_weight"], model["conv1_bias"], model["conv1_stride"], parms)
    conv1_convolution_end_time = time.time()
    conv1_convolution_time = conv1_convolution_end_time - conv1_convolution_start_time
    print(f'Conv1 Convolution time: {conv1_convolution_time:.4f} s')

    return 0


def encrypted_model(parms):

    PATH = f'Models/CNN_B/CBA_ConvNet_100.pth'
    load_model = torch.load(PATH)
    model = ConvNet()
    print(
        f'{model}'
    )
    model.load_state_dict(load_model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_data = datasets.MNIST('data', train=False, download=False, transform=transform)
    data = torch.utils.data.RandomSampler(test_data, replacement=True, num_samples=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, sampler=data)

    # initiate model, enc_model -> dictionary
    enc_model = init(model, parms)
    criterion = torch.nn.CrossEntropyLoss()
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    start_time = time.time()
    pbar = tqdm(test_loader)
    # enc_model.eval()
    for j, (data, target) in enumerate(pbar):
        # Encoding and encryption
        x = []
        for i in range(data.shape[1]):
            x.append(parms['encryptor'].encrypt(
                parms['encoder'].encode(data[0][i].flatten().tolist(), parms['scale'])))

        # Encryption evaluation
        enc_output = multiprocessing_model(x, enc_model, parms)
        #enc_output = enc_model(x)

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
        print(
            f'Temporary Accuracy:  {100 * np.sum(class_correct) / np.sum(class_total)}% ({int(np.sum(class_correct))}/{int(np.sum(class_total))})')
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

    train_dataset = torchvision.datasets.MNIST('data', train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.MNIST('data', train=False, transform=transform, download=False)
    batch_size = 100
    learning_rate = 0.001

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # data = torch.utils.data.RandomSampler(test_dataset, replacement=True, num_samples=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=4)  # ,  sampler=data)
    net = ConvNet()
    print(f'{net}')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    net = train(net, train_loader, optimizer, criterion)
    test(net, test_loader, criterion)


if __name__ == '__main__':
    # raw()
    parms = Encryption_Parameters()

    encrypted_model(parms)


