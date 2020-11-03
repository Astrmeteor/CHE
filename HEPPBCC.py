from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Convolution2D,MaxPooling2D,Flatten
import numpy as np

# keras中的mnist数据集已经被划分成了60,000个训练集，10,000个测试集的形式，按以下格式调用即可
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#############data pre_precessing#################
# 后端使用tensorflow时，即tf模式下，
# 会将100张RGB三通道的16*32彩色图表示为(100,16,32,3)，
# 第一个维度是样本维，表示样本的数目，
# 第二和第三个维度是高和宽，
# 最后一个维度是通道维，表示颜色通道数

x_train = np.loadtxt(open("encrypted_fm_train_images.csv", "rb"), delimiter=",", skiprows=0)
x_train = np.delete(x_train,-1,axis=1)
x_train = x_train.reshape(x_train.shape[0],28,28,1)

#x_train = x_train.reshape(x_train.shape[0],28,28,1)


x_test = np.loadtxt(open("encrypted_fm_test_images.csv", "rb"), delimiter=",", skiprows=0)
x_test = np.delete(x_test,-1,axis=1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

# 将X_train, X_test的数据格式转为float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 归一化可以加快梯度下降速度
#x_train /= 255
#x_test /= 255

# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵，
# 相当于将向量用one-hot重新编码
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


#################modeling#######################
# 建立序贯模型
model = Sequential()                                           #28*28*1

# 卷积层，对二维输入进行滑动窗卷积
# 当使用该层为第一层时，应提供input_shape参数，在tf模式中，通道维位于第三个位置
# padding：补0策略，为“valid”, “same” 。
# “valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
# 还是补成相同像素，或者是补1
model.add(Convolution2D(                                       #26*26*25
    filters=25,
    kernel_size=(3,3),
    padding='valid',
    input_shape=(28,28,1))) #通道数在后

# 池化层，选用Maxpooling，给定pool_size
model.add(MaxPooling2D(                                       #13*13*25
    pool_size=(2,2),
    strides=2))  #默认strides值为pool_size

# 卷积层
model.add(Convolution2D(                                      #11*11*50
    filters=50,
    kernel_size=(3,3),
    padding='valid'))

# 池化层
model.add(MaxPooling2D(pool_size=(2,2)))                      #5*5*50


# Flatten层，把多维输入进行一维化，常用在卷积层到全连接层的过渡
model.add(Flatten())                                           #1250

#包含100个神经元的全连接层，激活函数为ReLu，dropout比例为0.5
model.add(Dense(units=100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 包含10个神经元的输出层，激活函数为Softmax
model.add(Dense(units=10))
model.add(Activation('softmax'))

# 输出模型的参数信息
model.summary()

#######################cconfiguration############
# 配置模型的学习过程
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#######################training###################
model.fit(x_train,y_train,batch_size=100,epochs=10)

#######################evaluate###################
score=model.evaluate(x_test,y_test)
print('Test accuracy:', score[1])