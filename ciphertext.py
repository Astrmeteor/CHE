import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from Models.LeNet import LeNet
from Models.LeNet import EncLeNet
import tenseal as ts

def mnist_ciphertext():
    device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_dataset = datasets.MNIST('data', train=False, transform=transform, download=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    # encryption parameters
    bits_scale = 26
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )
    context.global_scale = pow(2, bits_scale)
    context.generate_galois_keys()

    model = LeNet()
    model.to(device)
    PATH = 'Parameters/mnist_lenet_10.pth'
    load_model = torch.load(PATH)
    model.load_state_dict(load_model)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    enc_model = EncLeNet(model)
    kernel_shape = model.conv1.kernel_size
    stride = model.conv1.stride[0]
    enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)


if __name__ == '__main__':
    mnist_ciphertext()