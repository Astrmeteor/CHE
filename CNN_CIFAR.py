import sys
import time
from keras.datasets import cifar10
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model as plot


# 开始下载数据集
t0 = time.time()  # 打开深度学习计时器
# CIFAR10 图片数据集
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32×32

X_train = X_train.astype('float32')  # uint8-->float32
X_test = X_test.astype('float32')
X_train /= 255  # 归一化到0~1区间
X_test /= 255
print('训练样例:', X_train.shape, Y_train.shape,
      ', 测试样例:', X_test.shape, Y_test.shape)


nb_classes = 10  # label为0~9共10个类别
# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
print("取数据耗时: %.2f seconds ..." % (time.time() - t0))


###################
# 1. 建立CNN模型
###################
print("开始建模CNN ...")
model = Sequential()  # 生成一个model
model.add(Convolution2D(
    32, 3, 3, border_mode='valid', input_shape=X_train.shape[1:]))  # C1 卷积层
model.add(Activation('relu'))  # 激活函数：relu, tanh, sigmoid

model.add(Convolution2D(32, 3, 3))  # C2 卷积层
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # S3 池化
model.add(Dropout(0.25))  #


model.add(Convolution2D(64, 3, 3, border_mode='valid')) # C4
model.add(Activation('relu'))


model.add(Convolution2D(64, 3, 3)) # C5
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))  # S6
model.add(Dropout(0.25))


model.add(Flatten())  # bottleneck 瓶颈
model.add(Dense(512))  # F7 全连接层, 512个神经元
model.add(Activation('relu'))  #
model.add(Dropout(0.5))


model.add(Dense(nb_classes))  # label为0~9共10个类别
model.add(Activation('softmax'))  # softmax 分类器
model.summary() # 模型小节
print("建模CNN完成 ...")




###################
# 2. 训练CNN模型
###################
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#plot(model, to_file='model1.png', show_shapes=True)  # 画模型图


model.fit(X_train, Y_train, batch_size=100, nb_epoch=50,
          validation_data=(X_test, Y_test))  # 81.34%, 224.08s
Y_pred = model.predict_proba(X_test, verbose=0)  # Keras预测概率Y_pred
print(Y_pred[:3, ])  # 取前三张图片的十类预测概率看看
score = model.evaluate(X_test, Y_test, verbose=0) # 评估测试集loss损失和精度acc
print('测试集 score(val_loss): %.4f' % score[0])  # loss损失
print('测试集 accuracy: %.4f' % score[1]) # 精度acc
print("耗时: %.2f seconds ..." % (time.time() - t0))