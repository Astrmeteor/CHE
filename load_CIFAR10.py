import pickle
import One_Hot_Encode_with_Keras
import numpy as np
import time
import EIVHE

def load_file(filename):
 with open(filename, 'rb') as fo:
    data = pickle.load(fo, encoding='latin1')
    return data

'''
x=5
CIFAR_data=np.ones((50000,3072))
for i in range (x):
     CIFAR_data[i*10000:(i+1)*10000,:] = np.array( load_file('dataset/CIFAR-10/data_batch_'+str(i+1)).get("data") )



#CIFAR_data = load_file('dataset/CIFAR-10/data_batch_1')
print(type(CIFAR_data.get('lables')))
lables=One_Hot_Encode_with_Keras.get_one_hot(CIFAR_data.get("lables"))
print(lables)
x=np.array( CIFAR_data.get("labels") )
w=16
e=EIVHE.EIVHE(CIFAR_data.get('data'),w)
print("明文：\n")
print(CIFAR_data.get('data'))

encrypt_start_time=time.time()
c,S=e.EIVHE_encrypt()
encrpt_end_time=time.time()
print("密文：\n")
print(c)
use_time=encrpt_end_time-encrypt_start_time
print("加密用时：%f"%use_time)

decrypt_start_time=time.time()
print("密文解密：\n")
print(e.EIVHE_decrypt(c, S, w))
decrypt_end_time=time.time()
use_time=decrypt_end_time-decrypt_start_time
print("解密用时：%f"%use_time)
'''