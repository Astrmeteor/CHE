import os
import struct
import numpy as np
import One_Hot_Encode_with_Keras
import time
import EIVHE


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


# path='dataset/MNIST'
# train_images,train_labels=load_mnist(path,kind='train') #train_images 60000*784 train_lable 1*60000
# test_images,test_labels=load_mnist(path,kind='t10k') #test_images 10000*784 test_lable 1*10000
# #train_labels=One_Hot_Encode_with_Keras.get_one_hot(train_labels)
# #print(train_labels)
# # #
# w=16
# e=EIVHE.EIVHE(test_images,train_images,w)
# # print("明文：\n")
# # print(train_images)
# #
# encrypt_start_time=time.time()
# c,S=e.EIVHE_encrypt()
# encrpt_end_time=time.time()
# print("密文：\n")
# print(c)
# use_time=encrpt_end_time-encrypt_start_time
# print("加密用时：%f"%use_time)
#
# decrypt_start_time=time.time()
# print("密文解密：\n")
# print(e.EIVHE_decrypt(c, S, w))
# decrypt_end_time=time.time()
# use_time=decrypt_end_time-decrypt_start_time
# print("解密用时：%f"%use_time)
