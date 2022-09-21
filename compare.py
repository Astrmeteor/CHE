import numpy as np
import matplotlib.pyplot as plt
import pickle
import pylab
import gzip
from tqdm import tqdm
from torchvision import datasets
import torch.utils.data as Data
import torchvision
import torch
import torchvision.transforms as transforms

def avg_pool():
    epochs = 100
    max_list = []
    avg_list = []
    max_pool_ratio_list = []
    max_avg_ratio_list = []
    for epoch in range(epochs):
        x = [np.random.randint(1, 100) for i in range(10)]
        # print(len(x))
        x_avg = np.mean(x)
        x_max = np.max(x)
        x_std = np.std(x)
        x_pool = x_avg + x_std
        max_pool_ratio = (x_max - x_pool) / x_max
        max_avg_ratio = (x_max - x_avg) / x_max

        if max_pool_ratio < 0:
            # print('{}: max pool ratio {}; max {}; average {}; std {}; pool {}'.format(epoch, max_pool_ratio, x_max, x_avg, x_std, x_pool))
            max_pool_ratio = 0
        # print(max_pool_ratio)
        # print(max_avg_ratio)
        max_list.append(x_max)
        avg_list.append(x_avg)
        max_pool_ratio_list.append(max_pool_ratio)
        max_avg_ratio_list.append(max_avg_ratio)

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(111)
    ax1.axis([0, 100, 0, 1])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Ratio', color='r')
    ax1.plot(range(epochs), max_pool_ratio_list, color='g')
    ax1.plot(range(epochs), max_avg_ratio_list, color='c')

    ax2 = ax1.twinx()
    ax2.axis([0, 100, 0, 100])
    ax2.set_ylabel('Value', color='b')
    ax2.plot(range(epochs), max_list, color='m')
    ax2.plot(range(epochs), avg_list, color='y')

    fig.legend(loc=1, bbox_to_anchor=(0.95, 0.95), borderaxespad=0,
               labels=['Ratio of Max-Mean-Std', 'Ratio of Max-Avg', 'Max', 'Average'])
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    plt.show()


def ratio_pool():
    epochs = 1000
    max_list = []
    avg_list = []
    times_pool_ratio_list = []
    max_avg_ratio_list = []
    efficiency = 0
    for epoch in range(epochs):
        x = [np.random.randint(1, 100) for i in range(10)]
        # print(len(x))
        x_avg = np.mean(x)
        x_max = np.max(x)

        x_pool = np.sum(x * np.random.dirichlet(np.ones(len(x)), size=1))
        max_pool_ratio = (x_max - x_pool) / x_max
        max_avg_ratio = (x_max - x_avg) / x_max

        # print(max_pool_ratio)
        # print(max_avg_ratio)
        max_list.append(x_max)
        avg_list.append(x_avg)
        times_pool_ratio_list.append(max_pool_ratio)
        max_avg_ratio_list.append(max_avg_ratio)
        if max_pool_ratio < max_avg_ratio:
            efficiency += 1
    efficiency_ratio = round(efficiency/epochs, 4)
    #print('Efficiency:{}% Note: the higher value, the more efficient'.format(efficiency_ratio * 100))
    # if bigger than average, hold; else, set value as average, we can see around half of

    # distribution ??? uniform gaussian???

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(111)
    ax1.axis([0, 100, 0, 1])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Ratio', color='r')
    ax1.plot(range(epochs), times_pool_ratio_list, color='g')
    ax1.plot(range(epochs), max_avg_ratio_list, color='c')

    ax2 = ax1.twinx()
    ax2.axis([0, 100, 0, 100])
    ax2.set_ylabel('Value', color='b')
    ax2.plot(range(epochs), max_list, color='m')
    ax2.plot(range(epochs), avg_list, color='y')

    fig.legend(loc=1, bbox_to_anchor=(0.95, 0.95), borderaxespad=0,
               labels=['Ratio of Max-Stochastic_weighted_sum', 'Ratio of Max-Avg', 'Max', 'Average'])
    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    plt.show()

    # efficiency ratio in 1000 times is around 50%
    return efficiency_ratio


def show_image():
    data = datasets.MNIST('data', train=False, download=False,transform=torchvision.transforms.ToTensor())
    data_loader = Data.DataLoader(dataset=data, batch_size=100, shuffle=True)

    images, labels = next(iter(data_loader))

    final_images = torch.empty((10, 1, 28, 28))
    for i in range(10):
        for image, label in zip(images, labels):
            if i == label:
                final_images[i] = image
                break
    final_images = torch.tensor(final_images)

    avg_final = torch.nn.AdaptiveAvgPool2d(28)(final_images)

    avg_final = torchvision.utils.make_grid(avg_final, nrow=5)
    avg_final = avg_final.numpy().transpose(1, 2, 0)
    plt.figure()
    plt.imshow(avg_final)
    plt.show()

    # raw mnist 0 - 9
    images_example = torchvision.utils.make_grid(final_images, nrow=5)
    images_example = images_example.numpy().transpose(1, 2, 0)

    plt.figure()
    plt.imshow(images_example)

    plt.show()

def cifar_show():
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'Total']
    train_set = torchvision.datasets.CIFAR10(root='data'
                                             , train=True
                                             , download=False
                                             , transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
    i = 0
    for batch in train_loader:
        if (i == 0):
            images, labels = batch  # image.shape==torch.Size([1, 3, 32, 32])
            print(labels)
            i += 1
        else:
            continue
    #images = torch.squeeze(images)#torch.Size([3, 32, 32])#显示单张的时候用
    print(images.shape)  # torch.Size([10, 3, 32, 32])
    # 显示
    #images = images/2 + 0.5
    # raw
    plt.figure()
    grid = torchvision.utils.make_grid(images, nrow=5)
    plt.imshow(np.transpose(grid, (1, 2, 0)))

    # max
    max_final = torch.nn.AdaptiveMaxPool2d(16)(images)
    grid = torchvision.utils.make_grid(max_final, nrow=5)
    plt.figure()
    plt.imshow(np.transpose(grid, (1, 2, 0)))

    # avg
    avg_final = torch.nn.AdaptiveAvgPool2d(16)(images)
    grid = torchvision.utils.make_grid(avg_final, nrow=5)
    plt.figure()
    plt.imshow(np.transpose(grid, (1, 2, 0)))

    # avg + std

    plt.show()


def mean_std_pooling():
    return 0

if __name__ == '__main__':
    #avg_pool()
    #show_image()
    #cifar_show()
    #print(np.random.dirichlet(np.ones(10), size=1))
    ratio_pool()
