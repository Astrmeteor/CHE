import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import tenseal as ts
import time
from tqdm import tqdm
from Models.LeNet import LeNet

torch.manual_seed(73)

# model
class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(ConvNet, self).__init__()
        # self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=7, padding=0, stride=3)
        # self.fc1 = torch.nn.Linear(256, hidden)
        self.fc1 = torch.nn.Linear(196, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        # conv
        x = self.conv1(x)
        # square activation
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(3,3), stride=1, padding=0) # padding=1
        # flatten & fc1
        # x = x.view(-1, 256)
        x = x.view(-1, 196)
        x = self.fc1(x)
        # square activation
        x = F.relu(x)
        # output
        x = self.fc2(x)
        return x

# train
def train(model, train_loader, criterion, optimizer, n_epochs=10):
    model.train()
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # calculate average losses
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    model.eval()
    return model


def test(model, test_loader, criterion):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval()
    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        # prediction
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader)
    print(f'Test Loss:{test_loss:.6f}\n')

    for label in range(10):
        print(
            f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}%'
            f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
        )

    print(
        f'\bTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}%'
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    )


# Encrypted Evaluation
# encrypted network
class EncConvNet:
    def __init__(self, torch_nn):

        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()

        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

    def forward(self, enc_x, windows_nb):
        # conv layer
        enc_channels = []

        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)

        # enc_x = self.conv1(enc_x)
        # enc_x = enc_x.view(-1, 256)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        # square activation
        enc_x.square_()
        # fc2 layer
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# Encrypted data test
def enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    start_time = time.time()
    for data, target in tqdm(test_loader):
        # Encoding and encryption

        # x_enc = ts.ckks_tensor(context, data.reshape(28, 28))

        x_enc, windows_nb = ts.im2col_encoding(
            context, data.view(28, 28).tolist(), kernel_shape[0], kernel_shape[1], stride
        )

        # windows_nb = 64

        # Encryption evaluation
        enc_output = enc_model(x_enc, windows_nb)
        # enc_output = enc_model(x_enc)
        # Decryption of result
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)
        # compute loss
        loss = criterion(output, target)
        test_loss += loss.item()

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare prediction to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1
    end_time = time.time()
    inference = end_time - start_time
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
        f'Inference Time: {inference:.6f}s ; {inference / int(np.sum(class_total)) :.6f} s/per'
    )


def train_test_model():
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_data = datasets.MNIST('data', train=True, download=False, transform=transform)
    test_data = datasets.MNIST('data', train=False, download=False, transform=transform)

    # train_data = datasets.CIFAR10('data', train=True, download=False, transform=transform)
    # test_data = datasets.CIFAR10('data', train=True, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_data = datasets.CIFAR10('data', train=True, download=False, transform=transform)
    test_data = datasets.CIFAR10('data', train=False, download=False, transform=transform)
    data = torch.utils.data.RandomSampler(test_data, replacement=True, num_samples=10)
    batch_size = 64
    epochs = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)#, sampler=data)


    model = ConvNet()
    print(model)
    # PATH = f'Parameters/mnist_lenet_{epochs}.pth'
    # load model
    # model = LeNet()
    # load_model = torch.load(PATH)
    # model.load_state_dict(load_model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    model = train(model, train_loader, criterion, optimizer, epochs)

    test(model, test_loader, criterion)
    PATH = f'Parameters/TenSEAL/cifar10_{model.__class__.__name__}_{epochs}.pth'
    torch.save(model.state_dict(), PATH)


def encrypted_model():
    PATH = f'Parameters/mnist_ConvNet_100.pth'
    load_model = torch.load(PATH)
    model = ConvNet()
    print(
        f'ConvNet\n {model}'
        )
    model.load_state_dict(load_model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_data = datasets.MNIST('data', train=False, download=False, transform=transform)
    # test_data = datasets.CIFAR10('data', train=False, download=False, transform=transforms.ToTensor())
    data = torch.utils.data.RandomSampler(test_data, replacement=True, num_samples=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, sampler=data)
    kernel_shape = model.conv1.kernel_size
    stride = model.conv1.stride[0]

    # Encryption parameters
    bit_scales = 26
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bit_scales, bit_scales, bit_scales, bit_scales, bit_scales, bit_scales, 31]
    )
    context.global_scale = pow(2, bit_scales)
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    enc_model = EncConvNet(model)
    print(
        f'EncConvNet\n {enc_model}'
    )
    criterion = torch.nn.CrossEntropyLoss()
    enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)


if __name__ == '__main__':
    train_test_model()
    # encrypted_model()
