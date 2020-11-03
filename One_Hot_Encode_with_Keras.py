from numpy import array
from numpy import argmax
import numpy as np
from keras.utils import to_categorical
# define example
def get_one_hot(data):
    data = array(data)
    encoded = to_categorical(data,10).astype('int')
    # inverted = argmax(encoded[0])
    # print(inverted)
    #print(encoded.shape)
    return encoded

# x=np.arange(1,50001,1)
# print(x.shape)
# print(get_one_hot(x).shape)
# print(get_one_hot(x))
# import numpy as np
# def get_one_hot(y):
#     u = np.unique(y)
#     coords = dict()
#     for i, x in enumerate(u):
#         coords[str(x)] = i
#                                             # 建立 value 和 key 之间的反向映射，
#                                             # 字典键值对（key-value pairs）的数目，就是 y 中不重复元素的数目
#     y_one_hot = np.zeros((len(y), len(u)))
#     for i, label in enumerate(y):
#         y_one_hot[i, coords[str(label)]] = 1
#     return y_one_hot