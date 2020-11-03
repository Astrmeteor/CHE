# import EIVHE
# import numpy as np
# x = np.array([[0,1,3,5,7],[2,4,6,8,10],[9,11,13,15,17]])
# w=16
# e=EIVHE.EIVHE(x,w)
# c,S=e.EIVHE_encrypt()
# print("明文：\n")
# print(x)
# print("密文：\n")
# print(c)
# print("密文解密：\n")
# print(e.EIVHE_decrypt(c, S, w))
#
# print("密文加法：\n")
# print(c + c)
# print("密文加法解密：\n")
# print(e.EIVHE_decrypt(c + c, S, w))
#
# print("密文减法：\n")
# print(2 * c - c)
# print("密文减法解密：\n")
# print(e.EIVHE_decrypt(2 * c - c, S, w))
#
# print("密文乘法：\n")
# print(2 * c)
# print("密文乘法解密：\n")
# print(e.EIVHE_decrypt(2 * c, S, w))

import load_CIFAR10
import EIVHE
import numpy as np
import One_Hot_Encode_with_Keras
def main():
    x = 5
    CIFAR_data = np.ones((50000, 3072))
    CIFAR_label = np.ones(50000)
    for i in range(x):
        CIFAR_data[i * 10000:(i + 1) * 10000, :] = np.array(
            load_CIFAR10.load_file('dataset/CIFAR-10/data_batch_' + str(i + 1)).get("data"))
        CIFAR_label[i * 10000 : (i + 1) * 10000]=np.array( load_CIFAR10.load_file('dataset/CIFAR-10/data_batch_' + str(i + 1)).get("labels") )
    CIFAR_label = One_Hot_Encode_with_Keras.get_one_hot(CIFAR_data.get("lables"))


    print(0)








if __name__=="__main__":
    main()
