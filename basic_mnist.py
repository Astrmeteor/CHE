import argparse  # Python 命令行解析工具
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import EIVHE
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        '''
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        '''

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
        '''


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # using softmax function for last layer
        # loss = nn.CrossEntropyLoss(output, target)
        loss.backward()
        loss_ = loss.mean().item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy


def main1():
    '''
    Encrypt dataset every epoch
    :return:
    '''
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_train = transforms.Compose([
        transforms.ToTensor()
        # ,transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
        # ,transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    model = Net().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    start_time = time.time()
    Keys = {}
    shared_key_number = []
    accuracies = []

    for epoch in range(1, args.epochs + 1):

        '''
        In every epoch, we collect used keys
        '''

        trainset = datasets.MNIST('data', train=True, download=False, transform=transform_train)
        testset = datasets.MNIST('data', train=False, download=False, transform=transform_test)
        # Encryption pre-process
        batch_encryption = args.batch_size
        blocks = int(trainset.train_data.shape[0] / batch_encryption)

        dict_w = set()

        for block in tqdm(range(blocks), desc='Encryption Train_Data_Set'):
            w = np.random.randint(1, 100)

            # normal distribution
            # w = round(np.random.normal(0.5, 0.1) * 100)

            dict_w.add(w)
            x = trainset.train_data[block * batch_encryption: (block + 1) * batch_encryption].detach().numpy()
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[1]).T
            e = EIVHE.EIVHE(x, w)
            c, S = e.EIVHE_encrypt()
            trainset.train_data[block * batch_encryption: (block + 1) * batch_encryption] = torch.tensor(
                c.T.reshape([c.T.shape[0], 28, 28]))
        Keys['Epoch_{}_Training_random_keys'.format(epoch)] = dict_w

        batch_encryption = args.test_batch_size
        blocks = int(testset.test_data.shape[0] / batch_encryption)

        dict_w_test = set()
        for block in tqdm(range(blocks), desc='Encryption Test_Data_Set'):
            w = np.random.randint(1, 100)
            # w = random.choice(list(dict_w))

            # normal distribution
            # w = round(np.random.normal(0.5, 0.1) * 100)

            dict_w_test.add(w)
            x = testset.test_data[block * batch_encryption: (block + 1) * batch_encryption].detach().numpy()
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[1]).T
            e = EIVHE.EIVHE(x, w)
            c, S = e.EIVHE_encrypt()
            testset.test_data[block * batch_encryption: (block + 1) * batch_encryption] = torch.tensor(
                c.T.reshape([c.T.shape[0], 28, 28]))

        Keys['Epoch_{}_Testing_random_keys'.format(epoch)] = dict_w_test

        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=args.test_batch_size, shuffle=False, **kwargs)

        one_epoch_start = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        one_epoch_end = time.time()
        accuracy = test(args, model, device, test_loader)
        accuracies.append(accuracy)

        print("\nTraining time:{:.4f}s".format(one_epoch_end - one_epoch_start))

        print('Epoch_{}_Training random keys:\n{}'.format(epoch, dict_w))
        print('Epoch_{}_Training used number:{}'.format(epoch, len(dict_w)))
        print('Epoch_{}_Testing random keys:\n{}'.format(epoch, dict_w_test))
        print('Epoch_{}_Testing used number:{}'.format(epoch, len(dict_w_test)))
        intersection = dict_w & dict_w_test
        shared_key_number.append(len(intersection))
        print('Epoch_{}_Used keys:\n{}\nShared number:{}'.format(epoch, intersection, len(intersection)))

    end_time = time.time()
    torch.save(model.state_dict(), 'Second_net_parameters_{}.pkl'.format(batch_encryption))
    print("\nTotal time:{:.4f}s".format(end_time - start_time))

    # plt

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.plot3D(list(range(1, epoch + 1)), shared_key_number, accuracies)
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Shared key number')
    ax.set_zlabel('Test Accuracy')
    plt.savefig('Test.png')
    #plt.show()

    plt.figure(2)
    plt.plot(shared_key_number, accuracies)
    plt.plot(shared_key_number, accuracies, 'ro')
    plt.xlabel('Shared key number')
    plt.ylabel('Test Accuracy')
    plt.savefig('1.png')

    plt.figure(3)
    plt.plot(list(range(1, epoch + 1)), accuracies)
    plt.plot(list(range(1, epoch + 1)), accuracies, 'ro')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.savefig('2.png')

    plt.figure(4)
    plt.plot(list(range(1, epoch + 1)), shared_key_number)
    plt.plot(list(range(1, epoch + 1)), shared_key_number, 'ro')
    plt.xlabel('Epoch')
    plt.ylabel('Shared key number')
    plt.savefig('3.png')



def main():
    '''
    Enrypt data before training and testing, pre-process
    :return:
    '''
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_train = transforms.Compose([
        transforms.ToTensor()
        # ,transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
        # ,transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    # trainset = datasets.MNIST('data', train=True, download=False, transform=transform_train)
    # testset = datasets.MNIST('data', train=False, download=False, transform=transform_test)

    trainset = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10('data', train=False, download=True, transform=transform_test)

    '''
    # Encryption pre-process
    batch_encryption = 1000
    
    # blocks = int(trainset.train_data.shape[0] / batch_encryption)
    blocks = int(trainset.data.shape[0] / batch_encryption)

    dict_w = set()

    encryption_start = time.time()
    for block in tqdm(range(blocks), desc='Encryption Train_Data_Set'):
        w = np.random.randint(1, 100)
        # w = 16
        # normal distribution
        # w = round(np.random.normal(0.5, 0.1) * 100)

        dict_w.add(w)
        # x = trainset.train_data[block*batch_encryption: (block+1)*batch_encryption].detach().numpy()
        x = trainset.data[block * batch_encryption: (block + 1) * batch_encryption]

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[1]).T
        e = EIVHE.EIVHE(x, w)
        c, S = e.EIVHE_encrypt()

        # trainset.train_data[block*batch_encryption: (block+1)*batch_encryption] = torch.tensor(c.T.reshape([c.T.shape[0],28,28]))
        trainset.data[block * batch_encryption: (block + 1) * batch_encryption] = torch.tensor(
            c.T.reshape([c.T.shape[0], 32, 32]))
    encryption_end = time.time()
    print("Encryption time is {}".format(encryption_end - encryption_start))

    torch.save(trainset, 'trainset_{}.pt'.format(batch_encryption))

    batch_encryption = 1

    # blocks = int(testset.test_data.shape[0] / batch_encryption)
    blocks = int(testset.data.shape[0] / batch_encryption)

    dict_w_test = set()
    for block in tqdm(range(blocks), desc='Encryption Test_Data_Set'):
        w = np.random.randint(1, 100)
        # w = random.choice(list(dict_w))
        # w = 30

        # normal distribution
        # w = round(np.random.normal(0.5, 0.1) * 100)

        # half normal distribution
        #w = 0
        # while w < 50:
        #    w = round(np.random.normal(0.75, 0.1) * 100)

        dict_w_test.add(w)

        # x = testset.test_data[block * batch_encryption: (block + 1) * batch_encryption].detach().numpy()
        x = testset.data[block * batch_encryption: (block + 1) * batch_encryption]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[1]).T
        e = EIVHE.EIVHE(x, w)
        c, S = e.EIVHE_encrypt()

        # testset.test_data[block * batch_encryption: (block + 1) * batch_encryption] = torch.tensor(c.T.reshape([c.T.shape[0], 28, 28]))
        testset.test_data[block * batch_encryption: (block + 1) * batch_encryption] = torch.tensor(c.T.reshape([c.T.shape[0], 32, 32]))
    torch.save(testset, 'testset_{}.pt'.format(batch_encryption))
    '''
    trainset = torch.load('trainset_one_same.pt')
    testset = torch.load('testset.pt')

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        one_epoch_start = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        one_epoch_end = time.time()
        test(args, model, device, test_loader)
        print("\nTraining time:{:.4f}s".format(one_epoch_end - one_epoch_start))
    end_time = time.time()
    torch.save(model.state_dict(), 'net_parameters_{}.pkl'.format(batch_encryption))
    print("\nTotal time:{:.4f}s".format(end_time - start_time))

    print('Training random keys:\n{}'.format(dict_w))
    print('Training used number:{}'.format(len(dict_w)))
    print('Testing random keys:\n{}'.format(dict_w_test))
    print('Testing used number:{}'.format(len(dict_w_test)))
    intersection = dict_w & dict_w_test
    print('Used keys:\n{}\nShared number:{}'.format(intersection, len(intersection)))


def cifar10_main():
    '''
    Enrypt data before training and testing, pre-process
    :return:
    '''
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_train = transforms.Compose([
        transforms.ToTensor()
        # ,transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
        # ,transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    trainset = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10('data', train=False, download=True, transform=transform_test)

    # Encryption pre-process
    batch_encryption = 1000

    blocks = int(trainset.data.shape[0] / batch_encryption)

    dict_w = set()

    encryption_start = time.time()
    for block in tqdm(range(blocks), desc='Encryption Train_Data_Set'):
        w = np.random.randint(1, 100)
        # w = 16
        # normal distribution
        # w = round(np.random.normal(0.5, 0.1) * 100)

        dict_w.add(w)

        x = trainset.data[block * batch_encryption: (block + 1) * batch_encryption]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).T
        e = EIVHE.EIVHE(x, w)
        c, S = e.EIVHE_encrypt()

        trainset.data[block * batch_encryption: (block + 1) * batch_encryption] = torch.tensor(
            c.T.reshape([c.T.shape[0], 32, 32, 3]))
    encryption_end = time.time()
    print("Encryption time is {}".format(encryption_end - encryption_start))

    #torch.save(trainset, 'trainset_{}.pt'.format(batch_encryption))

    batch_encryption = 1

    # blocks = int(testset.test_data.shape[0] / batch_encryption)
    blocks = int(testset.data.shape[0] / batch_encryption)

    dict_w_test = set()
    for block in tqdm(range(blocks), desc='Encryption Test_Data_Set'):
        w = np.random.randint(1, 100)
        # w = random.choice(list(dict_w))
        # w = 30

        # normal distribution
        # w = round(np.random.normal(0.5, 0.1) * 100)

        # half normal distribution
        # w = 0
        # while w < 50:
        #    w = round(np.random.normal(0.75, 0.1) * 100)

        dict_w_test.add(w)

        x = testset.data[block * batch_encryption: (block + 1) * batch_encryption]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).T
        e = EIVHE.EIVHE(x, w)
        c, S = e.EIVHE_encrypt()

        testset.data[block * batch_encryption: (block + 1) * batch_encryption] = torch.tensor(
            c.T.reshape([c.T.shape[0], 32, 32,3]))
    #torch.save(testset, 'testset_{}.pt'.format(batch_encryption))

    # trainset = torch.load('trainset_one_same.pt')
    # testset = torch.load('testset.pt')

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        one_epoch_start = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        one_epoch_end = time.time()
        test(args, model, device, test_loader)
        print("\nTraining time:{:.4f}s".format(one_epoch_end - one_epoch_start))
    end_time = time.time()
    torch.save(model.state_dict(), 'net_parameters_{}.pkl'.format(batch_encryption))
    print("\nTotal time:{:.4f}s".format(end_time - start_time))

    print('Training random keys:\n{}'.format(dict_w))
    print('Training used number:{}'.format(len(dict_w)))
    print('Testing random keys:\n{}'.format(dict_w_test))
    print('Testing used number:{}'.format(len(dict_w_test)))
    intersection = dict_w & dict_w_test
    print('Used keys:\n{}\nShared number:{}'.format(intersection, len(intersection)))


def validation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load('net_parameters_1000.pkl'))
    model = model.to(device)
    validate_image = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_path = 'Image/1.jpg'

    image = Image.open(image_path)

    image = image.convert('L').resize((28, 28), Image.ANTIALIAS)
    Image._show(image)
    image = np.array(image)
    w = np.random.randint(1, 100)
    e = EIVHE.EIVHE(image, w)
    c, S = e.EIVHE_encrypt()
    image = Image.fromarray(c)
    Image._show(image)

    imgblob = validate_image(image).unsqueeze(0)
    imgblob = torch.autograd.Variable(imgblob)
    torch.no_grad()
    output = model(imgblob)
    predict = torch.autograd.Variable(F.softmax(output, dim=1)).numpy()
    predict = np.argmax(predict).item()
    print(predict)


if __name__ == '__main__':
    # cifar10_main()
    #validation()
    # m =Net()
    # print(m)
    main()
