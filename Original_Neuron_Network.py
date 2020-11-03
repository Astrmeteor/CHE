import numpy as np
import struct
import random

TRAIN_ITEMS = 60000
TEST_ITEMS = 10000


def loadMnistData():
    mnist_data = []
    for img_file, label_file, items in zip(
            ['dataset/MNIST/train-images-idx3-ubyte', 'dataset/MNIST/t10k-images-idx3-ubyte'],
            ['dataset/MNIST/train-labels-idx1-ubyte', 'dataset/MNIST/t10k-labels-idx1-ubyte'],
            [TRAIN_ITEMS, TEST_ITEMS]):
        data_img = open(img_file, 'rb').read()
        data_label = open(label_file, 'rb').read()
        # fmt of struct unpack, > means big endian, i means integer, well, iiii mean 4 integers
        fmt = '>iiii'
        offset = 0
        magic_number, img_number, height, width = struct.unpack_from(fmt, data_img, offset)
        print('magic number is {}, image number is {}, height is {} and width is {}'.format(magic_number, img_number,
                                                                                            height, width))
        # slide over the 2 numbers above
        offset += struct.calcsize(fmt)
        # 28x28
        image_size = height * width
        # B means unsigned char
        fmt = '>{}B'.format(image_size)
        # because gemfield has insufficient memory resource
        if items > img_number:
            items = img_number
        images = np.empty((items, image_size))
        for i in range(items):
            images[i] = np.array(struct.unpack_from(fmt, data_img, offset))
            # 0~255 to 0~1
            images[i] = images[i] / 256
            offset += struct.calcsize(fmt)

        # fmt of struct unpack, > means big endian, i means integer, well, ii mean 2 integers
        fmt = '>ii'
        offset = 0
        magic_number, label_number = struct.unpack_from(fmt, data_label, offset)
        print('magic number is {} and label number is {}'.format(magic_number, label_number))
        # slide over the 2 numbers above
        offset += struct.calcsize(fmt)
        # B means unsigned char
        fmt = '>B'
        # because gemfield has insufficient memory resource
        if items > label_number:
            items = label_number
        labels = np.empty(items)
        for i in range(items):
            labels[i] = struct.unpack_from(fmt, data_label, offset)[0]
            offset += struct.calcsize(fmt)

        mnist_data.append((images, labels.astype(int)))

    return mnist_data


def vectorizedResult(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def loadMnistDataForNN():
    tr_d, te_d = loadMnistData()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorizedResult(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (list(training_data), list(test_data))

class GemfieldNeuronNetwork(object):
    def __init__(self, size_list):
        #size_list will be, e.g. [784, 30, 10]
        self.num_layers = len(size_list)
        self.size_list = size_list
        self.biases = [np.random.randn(y, 1) for y in size_list[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(size_list[:-1], size_list[1:])]

def sigmoid(z):
    """The sigmoid function."""
    rc = 1.0/(1.0+np.exp(-z))
    return rc

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def feedforward(self, a):
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a)+b)
    return a

def SGD(self, training_data, test_data, epochs, mini_batch_size, eta):
    n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [ training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)
        print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))


def update_mini_batch(self, mini_batch, eta):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

