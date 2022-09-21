import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import time
from tqdm import tqdm, trange

from Models.LeNet import LeNet
from Models.AlexNet import AlexNet
from Models.Conv3FC import Conv3FC

from PIL import Image

from Models.Layers.PoolingLayers import mean_pooling


def mnist_plaintext():
    device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = torchvision.datasets.MNIST('data', train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.MNIST('data', train=False, transform=transform, download=False)
    batch_size = 1000
    learning_rate = 0.001
    epochs = 100
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    net = LeNet()
    # net = AlexNet()
    # net = Conv3FC()
    net.to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    '''
    start_time = time.time()
    # training
    net.train()
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
    stop_time = time.time()
    print("Training time: {}s".format(stop_time - start_time))

    # save model
    
    torch.save(net.state_dict(), PATH)
    '''
    PATH = f'Parameters/mnist_{net.__class__.__name__}_{epochs}.pth'
    # load model
    load_model = torch.load(PATH)
    net.load_state_dict(load_model)

    # Testing
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set loss: {} ; Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def cifar10_pliantext():
    device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.CIFAR10('data', train=False, transform=transform, download=False)
    batch_size = 1000
    learning_rate = 0.0001
    epochs = 100
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    net = LeNet(input_channel=3, padding_num=0)
    # net = AlexNet()
    # net = Conv3FC()
    net.to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    start_time = time.time()
    # training
    net.train()
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
    stop_time = time.time()
    print("Training time: {}s".format(stop_time - start_time))

    # save model
    PATH = f'./Parameters/cifar10_lenet_{epochs}_{batch_size}.pth'
    torch.save(net.state_dict(), PATH)
    # load model
    # load_model = torch.load(PATH)
    # net.load_state_dict(load_model)

    # Testing
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set loss: {} ; Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def show_one_fig():
    img = Image.open('Image/test.jpeg')

    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)

    img_avg = torch.nn.AdaptiveAvgPool2d((500, 487))(img_tensor)
    # img_avg = pool_torch(img_tensor)

    img_avg_to_np = img_avg.numpy().squeeze()

    img = np.transpose(img_avg_to_np, (1, 2, 0))
    plt.figure()
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # mnist_plaintext()
    cifar10_pliantext()
    # show_one_fig()
