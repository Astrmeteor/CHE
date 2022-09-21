import torch
import torchvision
from torchvision import datasets
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision.transforms as transforms
import tenseal as ts
import numpy as np
from time import time


class FC_net(nn.Module):
    def __init__(self):
        super(FC_net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def train_fc_net(train_loader, epochs, criterion):

    model = FC_net()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view(x.size(0), 784)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # calculate average losses
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

        # save model
        PATH = f'./Parameters/FC_Net/plaintext.pth'
        torch.save(model.state_dict(), PATH)
    return model


def test_fc_net(model, test_loader, criterion):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval()
    for data, target in test_loader:
        data = data.view(data.size(0), 784)
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


class Enc_FC_net:
    def __init__(self, fc_net):
        self.weight_1 = fc_net.fc1.weight.data.tolist()[0]
        self.bias_1 = fc_net.fc1.bias.data.tolist()
        self._delta_weight_1 = 0
        self._delta_bias_1 = 0

        self.weight_2 = fc_net.fc2.weight.data.tolist()[0]
        self.bias_2 = fc_net.fc2.bias.data.tolist()
        self._delta_weight_2 = 0
        self._delta_bias_2 = 0

        self.weight_3 = fc_net.fc3.weight.data.tolist()[0]
        self.bias_3 = fc_net.fc3.bias.data.tolist()
        self._delta_weight_3 = 0
        self._delta_bias_3 = 0

        self.enc_out = []

    def forward(self, enc_x):  # forward MD: 3
        fc1 = enc_x.dot(self.weight_1) + self.bias_1
        self.enc_out.append(fc1)
        fc2 = fc1.dot(self.weight_2) + self.bias_2
        self.enc_out.append(fc2)
        fc3 = fc2.dot(self.weight_3) + self.bias_3
        self.enc_out.append(fc3)
        return fc3

    def backward(self, enc_x, enc_y):

        dz = self.enc_out[2] - enc_y
        dz = dz.mul(self.enc_out[2])
        # dz = dz.sum() * 0.1  # / 10
        self._delta_weight_3 += self.enc_out[1] * dz
        self._delta_bias_3 += dz

        self._delta_weight_2 += self.enc_out[0] * self._delta_weight_3
        self._delta_bias_2 += self._delta_bias_3

        self._delta_weight_1 += enc_x * self._delta_weight_2
        self._delta_bias_1 += self._delta_bias_2

    def update_parameters(self):
        learning_rate = 0.05
        self.weight_3 -= self._delta_weight_3 * learning_rate
        self.bias_3 -= self._delta_bias_3 * learning_rate

        self.weight_2 -= self._delta_weight_2 * learning_rate
        self.bias_2 -= self._delta_bias_2 * learning_rate

        self.weight_1 -= self._delta_weight_1 * learning_rate
        self.bias_1 -= self._delta_bias_1 * learning_rate

        # reset gradients
        self._delta_weight_1 = 0
        self._delta_bias_1 = 0

        self._delta_weight_2 = 0
        self._delta_bias_2 = 0

        self._delta_weight_3 = 0
        self._delta_bias_3 = 0

    def parameter_encrypt(self, context):
        
        self.weight_1 = ts.ckks_vector(context, self.weight_1)
        self.bias_1 = ts.ckks_vector(context, self.bias_1)

        self.weight_2 = ts.ckks_vector(context, self.weight_2)
        self.bias_2 = ts.ckks_vector(context, self.bias_2)

        self.weight_3 = ts.ckks_vector(context, self.weight_3)
        self.bias_3 = ts.ckks_vector(context, self.bias_3)
        
    def parameter_decrypt(self):
        self.weight_1 = self.weight_1.decrypt()
        self.bias_1 = self.bias_1.decrypt()

        self.weight_2 = self.weight_2.decrypt()
        self.bias_2 = self.bias_2.decrypt()

        self.weight_3 = self.weight_3.decrypt()
        self.bias_3 = self.bias_3.decrypt()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def encryption_context():
    bit_scales = 20
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=4096,
        coeff_mod_bit_sizes=[20, bit_scales, bit_scales, bit_scales, 20]  #
    )
    context.global_scale = pow(2, bit_scales)
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    return context


def train_enc_fc_net(train_loader, fc_net, epochs):
    # times = []
    context = encryption_context()  # try single context

    enc_fc_net = Enc_FC_net(fc_net)
    enc_fc_net.parameter_encrypt(context)
    t_start = time()
    for epoch in range(epochs):

        encryption_time = []
        train_time = []

        train_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        for batch_ix, (data, target) in enumerate(train_loader):

            # encryption data and target
            encrypt_start = time()
            # enc_x_train = [ts.ckks_vector(context, x.view(28*28).tolist()) for x in data]
            enc_x_train = [ts.ckks_tensor(context, x.view(28 * 28).tolist()) for x in data]
            target_one_hot = F.one_hot(target, num_classes=10)
            # enc_y_train = [ts.ckks_vector(context, y.view(-1).tolist()) for y in target_one_hot]
            enc_y_train = [ts.ckks_tensor(context, y.view(-1).tolist()) for y in target_one_hot]

            encrypt_end = time()
            encryption_time.append(encrypt_end-encrypt_start)  # encryption time of every batch in every epoch

            # train neural network over encrypted data
            train_start = time()
            output = []
            for enc_x, enc_y in zip(enc_x_train, enc_y_train):
                enc_out = enc_fc_net.forward(enc_x)
                enc_fc_net.backward(enc_x, enc_y)
                output.append(enc_out.decrypt())
            enc_fc_net.update_parameters()
            train_end = time()
            train_time.append(train_end-train_start)

            # compute accuracy after decrypt (over plaintext)
            # output = enc_out_all.decrypt()
            output = torch.tensor(output).view(1, -1)
            # compute loss
            loss = criterion(output, target)
            train_loss += loss.item()

            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare prediction to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            label = target.data[0]
            class_correct[label] += correct.item()
            class_total[label] += 1

        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Encryption Time: {} Train Time: {} Training Loss: {:.6f}'.format(epoch, sum(train_time), sum(encryption_time), train_loss))
    t_end = time()


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    batch_size = 1000
    train_data = datasets.MNIST('data', train=True, download=False, transform=transform)
    test_data = datasets.MNIST('data', train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()

    # train plaintext model and test
    # model = train_fc_net(train_loader, epochs=100, criterion=criterion)
    # test_fc_net(model, test_loader, criterion=criterion) # plaintext accuracy 89.62%

    # load trained model and train ciphertext and test
    PATH = f'./Parameters/FC_Net/plaintext.pth'
    load_model = torch.load(PATH)
    model = FC_net()
    model.load_state_dict(load_model)
    print(model.parameters())
    train_enc_fc_net(train_loader, model, epochs=10)


