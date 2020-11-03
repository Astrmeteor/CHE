import argparse
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import load_CIFAR10
from tensorflow.examples.tutorials.mnist import input_data
import EIVHE
import load_mnist
import One_Hot_Encode_with_Keras

"""
类型：抽象类
说明：规则化数据接口，一般用于数据预处理中。
"""


class interface_normalize_data(object):
    """
    类型：公有成员变量
    说明：规则化数据过程中，定义一个无穷小精度，用来防止数据计算中的非法操作。
    """
    epsilon = 1e-8

    """
    类型：抽象公有成员函数
    说明：用来规则化数据。
    参数：
        data -- 待处理的数据。
    返回值：
        data -- 处理后的数据。
    """

    def normalize_data(self, data):
        pass


"""
类型：实体类，继承自抽象类interface_normalize_data
说明：用于中心化数据，使数据中心在坐标原点上。
"""


class mean_normalization(interface_normalize_data):

    def normalize_data(self, data):
        # 计算数据每个维度的期望，并用每一条数据减去期望。
        data = data - np.mean(data, axis=1, keepdims=True)
        return data


"""
类型：实体类，继承自抽象类interface_normalize_data
说明：用于中心化数据，并除以方差，使数据中心在坐标原点上，并且使每个维度之间的跨度相似。
"""


class variance_normalization(interface_normalize_data):

    def normalize_data(self, data):
        data = data - np.mean(data, axis=1, keepdims=True)
        # 计算数据每个维度的方差。
        variance = np.mean(np.square(data), axis=1, keepdims=True)
        # 除以方差并在除数上加上无穷小精度。
        data = data / (variance + self.epsilon)
        return data


"""
类型：实体类，继承自抽象类interface_normalize_data
说明：用于Z-Score统计，与上述实体类的区别是除以标准差而不是方差。
"""


class zscore_normalization(interface_normalize_data):

    def normalize_data(self, data):
        data = data - np.mean(data, axis=1, keepdims=True)
        variance = np.mean(np.square(data), axis=1, keepdims=True)
        # 除以标准差并在除数上加上无穷小精度。
        data = data / np.sqrt(variance + self.epsilon)
        return data


"""   
类型：抽象类
说明：神经网络初始化参数接口。
"""


class interface_initialize_parameters(object):
    """
    类型：公有成员变量
    说明：用来定义输入层、隐藏层、输出层每层的神经元个数。
    """
    structure = None

    """
    类型：公有成员变量
    说明：随机种子，用来产生随机数。
    """
    seed = 1

    """
    类型：抽象公有成员函数
    说明：用来初始化参数。
    """

    def initialize_parameters(self):
        pass


"""
类型：实体类
说明：标准的x-avier参数初始化，继承自抽象类interface_initialize_parameters
"""


class xavier_initialize_parameters(interface_initialize_parameters):
    """
    类型：公有成员函数
    说明：用来初始化参数。
    参数：无
    返回值：
        parameters -- 返回初始化后的参数。
    """

    def initialize_parameters(self):
        np.random.seed(self.seed)
        parameters = {}

        # 初始化两类参数，一种是W1、W2、W3……，另一种是b1、b2、b3……。其中数字代表层数。
        # W的维度为(当前层神经元数，前一层神经元数)。b的维度为(当前层神经元数，1)。
        for l in range(1, len(self.structure)):
            parameters["W" + str(l)] = np.random.randn(self.structure[l], self.structure[l - 1]) / np.sqrt(
                self.structure[l - 1] / 2)
            parameters["b" + str(l)] = np.zeros((self.structure[l], 1))

        return parameters


"""
类型：实体类
说明：具有batch normalization功能的x-avier参数初始化，继承自抽象类interface_initialize_parameters
"""


class xavier_initialize_parameters_BN(interface_initialize_parameters):
    """
    类型：公有成员函数
    说明：用来初始化参数。
    参数：无
    返回值：
        parameters -- 返回初始化后的参数。
    """

    def initialize_parameters(self):
        np.random.seed(self.seed)
        parameters = {}

        # 因batch normalization需要，初始化三类参数，W1、W2、W3……，gamma1、gamma2、gamma3……，beta1、beta2、beta3……。其中数字代表层数。
        # W的维度为(当前层神经元数，前一层神经元数)。gamma与beta的维度均为(当前层神经元数，1)。
        for l in range(1, len(self.structure)):
            parameters["W" + str(l)] = np.random.randn(self.structure[l], self.structure[l - 1]) / np.sqrt(
                self.structure[l - 1] / 2)
            parameters["gamma" + str(l)] = np.ones((self.structure[l], 1))
            # parameters["gamma" + str(l)] = np.random.randn(self.structure[l], 1) / np.sqrt(self.structure[l]/2)
            parameters["beta" + str(l)] = np.zeros((self.structure[l], 1))

        return parameters


"""
类型：抽象类
说明：计算激活函数的值和激活函数的梯度值
"""


class interface_activation(object):
    """
    类型：公有成员变量
    说明：规则化数据过程中，定义一个无穷小精度，用来防止数据计算中的非法操作。
    """
    epsilon = 1e-8

    """
    类型：抽象公有成员函数
    说明：计算激活函数的值。
    """

    def activate_function(self, *arguments):
        pass

    """
    类型：抽象公有成员函数
    说明：计算激活函数的梯度值。
    """

    def derivation_activate_function(self, *arguments):
        pass


"""
类型：抽象类
说明：计算代价函数的值和代价函数的梯度值
"""


class interface_cost(object):
    """
    类型：公有成员变量
    说明：规则化数据过程中，定义一个无穷小精度，用来防止数据计算中的非法操作。
    """
    epsilon = 1e-8

    """
    类型：抽象公有成员函数
    说明：计算代价函数并返回代价值。
    """

    def cost_function(self, *arguments):
        pass

    """
    类型：抽象公有成员函数
    说明：计算代价函数的梯度并返回梯度值。
    """

    def derivation_cost_function(self, *arguments):
        pass


"""
类型：具体类
说明：relu激活函数，继承自interface_activation。
"""


class relu(interface_activation):
    """
    类型：公有成员函数
    说明：计算relu函数的值。
    参数：
        Z -- 每一层（不包括最后一层）的线性值。
    返回值：
        A -- 激活值。
    """

    def activate_function(self, *arguments):
        Z = arguments[0]

        A = np.maximum(Z, 0)

        return A

    """
    类型：公有成员函数
    说明：计算relu函数的梯度值。
    参数：
        dA -- 每一层（不包括最后一层）激活值的梯度值。
    返回值：
        A -- 线性值的梯度值。
    """

    def derivation_activate_function(self, *arguments):
        dA = arguments[0]
        A = arguments[1]

        dZ = dA * np.where(A > 0, 1, 0)

        return dZ


"""
类型：具体类
说明：softmax代价函数，继承自interface_activation。
"""


class softmax(interface_cost):
    """
    类型：公有成员函数
    说明：计算softmax代价函数的代价值与输出层（最后一层）的激活值。
    参数：
        ZL -- 输出层（最后一层）的线性值。
        Y -- 数据标签。
        loss -- 如果需要L1或L2正则化，则此参数计算了正则化的代价值。
    返回值：
        cost -- 代价值。
        AL -- 输出层（最后一层）的激活值。
    """

    def cost_function(self, *arguments):
        ZL = arguments[0]
        Y = arguments[1]
        loss = arguments[2]

        AL = np.exp(ZL) / (np.sum(np.exp(ZL), axis=0, keepdims=True) + self.epsilon)
        cost = -np.sum(Y * np.log(AL + self.epsilon)) / Y.shape[1] + loss
        return cost, AL

    """
    类型：公有成员函数
    说明：计算softmax代价函数的梯度值与输出层（最后一层）的线性梯度值。
    参数：
        AL -- 输出层（最后一层）的激活值。
        Y -- 数据标签。
    返回值：
        dZL -- 输出层（最后一层）的线性值的梯度值。
    """

    def derivation_cost_function(self, *arguments):
        AL = arguments[0]
        Y = arguments[1]

        dZL = (AL - Y) / Y.shape[1]
        return dZL


"""
类型：抽象类
说明：定义神经网络训练过程中所需要的规则化功能。
"""


class interface_regularization(object):
    """
    类型：公有成员变量
    说明：规则化数据过程中，定义一个无穷小精度，用来防止数据计算中的非法操作。
    """
    epsilon = 1e-8

    """
    类型：抽象公有成员函数
    说明：向前算法中所需要的规则化动作。
    """

    def forward_regularization(self, *arguments):
        pass

    """
    类型：抽象公有成员函数
    说明：计算代价函数中所需要的规则化动作。
    """

    def cost_regularization(self, *arguments):
        pass

    """
    类型：抽象公有成员函数
    说明：向后算法中所需要的规则化动作。
    """

    def backward_regularization(self, *arguments):
        pass


"""
类型：具体类
说明：batch normalization功能，继承自interface_regularization
"""


class batch_normlizer(interface_regularization):
    """
    类型：公有成员函数
    说明：batch normalization功能的向前算法。
    参数：
        Z -- 每一层的线性值。
    返回值：
        Zcenter -- Z的中心化计算，使期望位于坐标的原点。
        Zvariance -- Z的方差。
        Ztilde -- Z-score标准化后的Z数值。
    """

    def forward_regularization(self, *arguments):
        Z = arguments[0]

        Zmean = np.mean(Z, axis=1, keepdims=True)
        Zcenter = Z - Zmean
        Zvariance = np.mean(np.square(Zcenter), axis=1, keepdims=True)
        Ztilde = Zcenter / np.sqrt(Zvariance + self.epsilon)

        return Zcenter, Zvariance, Ztilde

    """
    类型：公有成员函数
    说明：batch normalization不需要此步骤，设置为空函数。
    """

    def cost_regularization(self, *arguments):
        pass

    """
    类型：公有成员函数
    说明：batch normalization功能的向后算法。
    参数：
        dZnorm -- 每一层batch normalization线性值的梯度值。
        gamma -- batch normalization的参数。
        Zcenter -- Z的中心化计算，使期望位于坐标的原点。
        Zvariance -- Z的方差。
    返回值：
        dZ -- 每一层线性值的梯度值。
    """

    def backward_regularization(self, *arguments):
        dZnorm = arguments[0]
        gamma = arguments[1]
        Zcenter = arguments[2]
        Zvariance = arguments[3]

        dZtilde = np.multiply(dZnorm, gamma)
        dZvariance = np.sum(np.multiply(np.multiply(dZtilde, Zcenter), np.power(Zvariance + self.epsilon, -3 / 2) / -2),
                            axis=1, keepdims=True)
        dZmean = np.sum(np.multiply(dZtilde, -1 / np.sqrt(Zvariance + self.epsilon)), axis=1, keepdims=True)
        dZ = np.multiply(dZtilde, 1 / np.sqrt(Zvariance + self.epsilon)) + np.multiply(dZvariance,
                                                                                       2 * Zcenter / Zcenter.shape[1])
        + np.multiply(dZmean, 1 / Zcenter.shape[1])
        return dZ


"""
类型：具体类
说明：dropout功能，继承自interface_regularization
"""


class dropout(interface_regularization):
    """
    类型：公有成员变量
    说明：以概率keep_prob随机删除当前层的神经元节点。
    """
    keep_prob = 1.

    """
    类型：公有成员函数
    说明：dropout向前算法，以概率keep_prob随机删除当前层的神经元节点。
    参数:
        A -- 当前层的激活值
    返回值：
        D -- 掩码矩阵。
        A -- 随机删除神经元节点后的激活值。
    """

    def forward_regularization(self, *arguments):
        A = arguments[0]

        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < self.keep_prob)
        A = np.multiply(A, D)
        A = A / self.keep_prob

        return D, A

    """
    类型：公有成员函数
    说明：dropout不需要此步骤，设置为空函数。
    """

    def cost_regularization(self, *arguments):
        pass

    """
    类型：公有成员函数
    说明：dropout向后算法，和向前算法相同，删除向前算法中对应的已删除神经元节点。
    参数：
        dA -- 当前层神经元的激活值的梯度值。
        D -- 掩码矩阵。
    返回值：
        dA -- 随机删除神经元节点后的激活值的梯度。
    """

    def backward_regularization(self, *arguments):
        dA = arguments[0]
        D = arguments[1]

        dA = np.multiply(dA, D)
        dA = dA / self.keep_prob

        return dA


"""
类型：具体类
说明：L2正则化功能，继承自interface_regularization
"""


class L2(interface_regularization):
    """
    类型：公有成员变量
    说明：设置参数lambda，决定权值规则化在代价函数中所占的比重。
    """
    lambd = 0.

    """
    类型：公有成员函数
    说明：L2规则化不需要此步骤，设置为空函数。
    """

    def forward_regularization(self, *arguments):
        pass

    """
    类型：公有成员函数
    说明：在代价函数中计算L2规则化的权重。
    参数：
        parameters -- 学习参数。
        layer_number -- 层数（包含输入层、隐藏层、输出层）。
        m -- 数据的数量。
    返回值：
        loss -- 代价函数中L2正则化所占的比重。
    """

    def cost_regularization(self, *arguments):
        parameters = arguments[0]
        layer_number = arguments[1]
        m = arguments[2]
        total = 0.

        for l in range(1, layer_number + 1):
            total += np.sum(np.square(parameters["W" + str(l)]))

        total /= 2 * m

        return self.lambd * total

    """
    类型：公有成员函数
    说明：在向后算法中计算L2规则化的权重。
    参数：
        W -- 当前层的学习参数。
        m -- 数据的数量。
    返回值：
        dloss -- 向后算法中L2规则化权重部分的梯度。
    """

    def backward_regularization(self, *arguments):
        W = arguments[0]
        m = arguments[1]
        return self.lambd * W / m


"""
类型：具体类
说明：L1正则化功能，继承自interface_regularization
"""


class L1(interface_regularization):
    """
    类型：公有成员变量
    说明：设置参数lambda，决定权值规则化在代价函数中所占的比重。
    """
    lambd = 0.

    """
    类型：公有成员函数
    说明：L1规则化不需要此步骤，设置为空函数。
    """

    def forward_regularization(self, *arguments):
        pass

    """
    类型：公有成员函数
    说明：在代价函数中计算L1规则化的权重。
    参数：
        parameters -- 学习参数。
        layer_number -- 层数（包含输入层、隐藏层、输出层）。
        m -- 数据的数量。
    返回值：
        loss -- 代价函数中L1正则化所占的比重。
    """

    def cost_regularization(self, *arguments):
        parameters = arguments[0]
        layer_number = arguments[1]
        m = arguments[2]
        total = 0.

        for l in range(1, layer_number + 1):
            total += np.sum(np.abs(parameters["W" + str(l)]))

        total /= m

        return self.lambd * total

    """
    类型：公有成员函数
    说明：在向后算法中计算L1规则化的权重。
    参数：
        W -- 当前层的学习参数。
        m -- 数据的数量。
    返回值：
        dloss -- 向后算法中L1规则化权重部分的梯度。
    """

    def backward_regularization(self, *arguments):
        W = arguments[0]
        m = arguments[1]
        # 此处求梯度利用绝对值求导。
        return self.lambd * np.sign(W) / m


"""
类型：抽象类
说明：定义向前向后算法。
"""


class interface_propagation(object):
    """
    类型：公有成员变量
    说明：定义一个空字典，用来存储规则化功能，可以存储多个规则化器regularizer。
    """
    regularizer = {}

    """
    类型：公有成员变量
    说明：规则化数据过程中，定义一个无穷小精度，用来防止数据计算中的非法操作。
    """
    epsilon = 1e-8

    """
    类型：公有成员变量
    说明：定义一个空字典，用来存储激活函数；激活函数的梯度函数；代价函数；代价函数的梯度函数。
    """
    functor = {}

    """
    类型：保护成员变量
    说明：定义一个空字典，用来记录在向前算法中产生的中间结果。
    """
    _intermediate_caches = {}

    """
    类型：抽象公有函数
    说明：向前算法。
    参数：
        training_set -- 训练集。
        training_label -- 训练标签。
        parameters -- 训练参数。
    """

    def forward_propagation(self, training_set, training_label, parameters):
        pass

    """
    类型：抽象公有函数
    说明：向后算法。
    参数：
        training_set -- 训练集。
        training_label -- 训练标签。
        parameters -- 训练参数。
    """

    def backward_propagation(self, training_set, training_label, parameters):
        pass


"""
类型：具体类
说明：标准的向前向后算法，继承自interface_propagation。
"""


class propagation_standard(interface_propagation):

    def forward_propagation(self, training_set, training_label, parameters):
        caches = self._intermediate_caches
        cost = None

        if (len(parameters) < 2):
            return cost, caches

            # 向前算法
        caches["A0"] = training_set
        Z = 0.

        layer_number = len(parameters) // 2
        for l in range(1, layer_number):
            Z = np.dot(parameters["W" + str(l)], caches["A" + str(l - 1)]) + parameters["b" + str(l)]
            # 激活函数
            caches["A" + str(l)] = self.functor["activation"].activate_function(Z)

            # 添加dropout正则化功能
            if ("dropout" in self.regularizer.keys()):
                caches["D" + str(l)], caches["A" + str(l)] = self.regularizer["dropout"].forward_regularization(
                    caches["A" + str(l)])

        Z = np.dot(parameters["W" + str(layer_number)], caches["A" + str(layer_number - 1)]) + parameters[
            "b" + str(layer_number)]

        # 添加L1、L2正则化功能
        loss = 0.
        if ("L2" in self.regularizer.keys()):
            loss += self.regularizer["L2"].cost_regularization(parameters, layer_number, training_set.shape[1])

        if ("L1" in self.regularizer.keys()):
            loss += self.regularizer["L1"].cost_regularization(parameters, layer_number, training_set.shape[1])

            # 计算代价函数
        cost, caches["A" + str(layer_number)] = self.functor["cost"].cost_function(Z, training_label, loss)

        return cost

    def backward_propagation(self, training_set, training_label, parameters):

        caches = self._intermediate_caches

        grad_parameters = {}

        # 向后算法。
        layer_number = len(parameters) // 2
        # 代价函数的梯度计算。
        dZ = self.functor["cost"].derivation_cost_function(caches["A" + str(layer_number)], training_label)
        for l in reversed(range(1, layer_number + 1)):
            grad_parameters["dW" + str(l)] = np.dot(dZ, caches["A" + str(l - 1)].T)

            # 添加L1、L2正则化功能。
            if ("L2" in self.regularizer.keys()):
                grad_parameters["dW" + str(l)] += self.regularizer["L2"].backward_regularization(
                    parameters["W" + str(l)], training_set.shape[1])

            if ("L1" in self.regularizer.keys()):
                grad_parameters["dW" + str(l)] += self.regularizer["L1"].backward_regularization(
                    parameters["W" + str(l)], training_set.shape[1])

            grad_parameters["db" + str(l)] = np.sum(dZ, axis=1, keepdims=True)
            if (l > 1):
                dA = np.dot(parameters["W" + str(l)].T, dZ)
                # 添加dropout正则化功能。
                if ("dropout" in self.regularizer.keys()):
                    dA = self.regularizer["dropout"].backward_regularization(dA, caches["D" + str(l - 1)])
                # 激活函数的梯度计算。
                dZ = self.functor["activation"].derivation_activate_function(dA, caches["A" + str(l - 1)])

        return grad_parameters


"""
类型：具体类
说明：带有Batch normalization的向前向后算法，继承自interface_propagation。
"""


class propagation_BN(interface_propagation):

    def forward_propagation(self, training_set, training_label, parameters):
        caches = self._intermediate_caches
        cost = None

        if (len(parameters) < 2):
            return cost, caches

        caches["A0"] = training_set
        Z = 0.

        # 注意，因为Batch normalization有三个学习参数，这里要除以3取整。
        layer_number = len(parameters) // 3
        for l in range(1, layer_number):
            Z = np.dot(parameters["W" + str(l)], caches["A" + str(l - 1)])

            # 向前算法中的Batch normalization步骤。
            caches["Zcenter" + str(l)], caches["Zvariance" + str(l)], caches["Ztilde" + str(l)] = self.regularizer[
                "batch_normlizer"].forward_regularization(Z)

            Z = np.multiply(parameters["gamma" + str(l)], caches["Ztilde" + str(l)]) + parameters["beta" + str(l)]
            caches["A" + str(l)] = self.functor["activation"].activate_function(Z)

            if ("dropout" in self.regularizer.keys()):
                caches["D" + str(l)], caches["A" + str(l)] = self.regularizer["dropout"].forward_regularization(
                    caches["A" + str(l)])

        Z = np.dot(parameters["W" + str(layer_number)], caches["A" + str(layer_number - 1)])
        caches["Zcenter" + str(layer_number)], caches["Zvariance" + str(layer_number)], caches[
            "Ztilde" + str(layer_number)] = self.regularizer["batch_normlizer"].forward_regularization(Z)

        Z = np.multiply(parameters["gamma" + str(layer_number)], caches["Ztilde" + str(layer_number)]) + parameters[
            "beta" + str(layer_number)]

        loss = 0.
        if ("L2" in self.regularizer.keys()):
            loss += self.regularizer["L2"].cost_regularization(parameters, layer_number, training_set.shape[1])

        if ("L1" in self.regularizer.keys()):
            loss += self.regularizer["L1"].cost_regularization(parameters, layer_number, training_set.shape[1])

        cost, caches["A" + str(layer_number)] = self.functor["cost"].cost_function(Z, training_label, loss)
        return cost


    def backward_propagation(self, training_set, training_label, parameters):
        caches = self._intermediate_caches

        grad_parameters = {}

        # 注意，因为Batch normalization有三个学习参数，这里要除以3取整。
        layer_number = len(parameters) // 3
        dZnorm = self.functor["cost"].derivation_cost_function(caches["A" + str(layer_number)], training_label)
        for l in reversed(range(1, layer_number + 1)):
            grad_parameters["dgamma" + str(l)] = np.sum(np.multiply(dZnorm, caches["Ztilde" + str(l)]), axis=1,
                                                        keepdims=True)
            grad_parameters["dbeta" + str(l)] = np.sum(dZnorm, axis=1, keepdims=True)

            # 向前算法中的Batch normalization步骤。
            dZ = self.regularizer["batch_normlizer"].backward_regularization(dZnorm, parameters["gamma" + str(l)],
                                                                             caches["Zcenter" + str(l)],
                                                                             caches["Zvariance" + str(l)])

            grad_parameters["dW" + str(l)] = np.dot(dZ, caches["A" + str(l - 1)].T)

            if ("L2" in self.regularizer.keys()):
                grad_parameters["dW" + str(l)] += self.regularizer["L2"].backward_regularization(parameters["W" + str(l)],
                                                                                                 training_set.shape[1])

            if ("L1" in self.regularizer.keys()):
                grad_parameters["dW" + str(l)] += self.regularizer["L1"].backward_regularization(parameters["W" + str(l)],
                                                                                                 training_set.shape[1])

            if (l > 1):
                dA = np.dot(parameters["W" + str(l)].T, dZ)
                if ("dropout" in self.regularizer.keys()):
                    dA = self.regularizer["dropout"].backward_regularization(dA, caches["D" + str(l - 1)])
                dZnorm = self.functor["activation"].derivation_activate_function(dA, caches["A" + str(l - 1)])
        return grad_parameters


"""
类型：抽象类
说明：定义优化算法。
"""


class interface_optimization(object):
    """
    类型：公有成员变量
    说明：规则化数据过程中，定义一个无穷小精度，用来防止数据计算中的非法操作。
    """
    epsilon = 1e-8

    """
    类型：公有成员变量
    说明：定义优化算法的学习率。
    """
    learning_rate = 1e-3

    """
    类型：抽象公有成员函数
    说明：定义优化算法的初始化工作。
    """

    def initialization(self, *arguments):
        pass

    """
    类型：抽象公有成员函数
    说明：定义优化算法的计算步骤。
    """

    def optimization(self, *arguments):
        pass


"""
类型：具体类
说明：标准的梯度下降算法。
"""


class gradient_descent_standard(interface_optimization):

    def initialization(self, *arguments):
        pass

    def optimization(self, *arguments):
        parameters = arguments[0]
        grads = arguments[1]

        L = len(parameters) // 2

        for l in range(1, L + 1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - self.learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - self.learning_rate * grads["db" + str(l)]

        return parameters


"""
类型：具体类
说明：带有Batch normalization的梯度下降算法。
"""


class gradient_descent_BN(interface_optimization):

    def initialization(self, *arguments):
        pass

    def optimization(self, *arguments):
        parameters = arguments[0]
        grads = arguments[1]

        L = len(parameters) // 3

        for l in range(1, L + 1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - self.learning_rate * grads["dW" + str(l)]
            parameters["gamma" + str(l)] = parameters["gamma" + str(l)] - self.learning_rate * grads["dgamma" + str(l)]
            parameters["beta" + str(l)] = parameters["beta" + str(l)] - self.learning_rate * grads["dbeta" + str(l)]

        return parameters


"""
类型：具体类
说明：标准的adam算法。
"""


class adam_standard(interface_optimization):
    __v = {}
    __s = {}
    __t = 0

    beta1 = 0.9
    beta2 = 0.999

    def initialization(self, *arguments):

        parameters = arguments[0]

        v = self.__v
        s = self.__s
        self.__t = 0

        L = len(parameters) // 2

        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)

            s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)

    def optimization(self, *arguments):

        parameters = arguments[0]
        grads = arguments[1]

        v = self.__v
        s = self.__s

        self.__t += 1

        v_corrected = {}
        s_corrected = {}

        L = len(parameters) // 2

        for l in range(1, L + 1):
            v["dW" + str(l)] = self.beta1 * v["dW" + str(l)] + (1 - self.beta1) * grads['dW' + str(l)]
            v["db" + str(l)] = self.beta1 * v["db" + str(l)] + (1 - self.beta1) * grads['db' + str(l)]

            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(self.beta1, self.__t))
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(self.beta1, self.__t))

            s["dW" + str(l)] = self.beta2 * s["dW" + str(l)] + (1 - self.beta2) * np.power(grads['dW' + str(l)], 2)
            s["db" + str(l)] = self.beta2 * s["db" + str(l)] + (1 - self.beta2) * np.power(grads['db' + str(l)], 2)

            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(self.beta2, self.__t))
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(self.beta2, self.__t))

            parameters["W" + str(l)] = parameters["W" + str(l)]
            - self.learning_rate * (v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + self.epsilon))
        parameters["b" + str(l)] = parameters["b" + str(l)]
        - self.learning_rate * (v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + self.epsilon))
        return parameters


"""
类型：具体类
说明：带有Batch normalization的adam算法。
"""


class adam_BN(interface_optimization):
    __v = {}
    __s = {}
    __t = 0

    beta1 = 0.9
    beta2 = 0.999

    def initialization(self, *arguments):

        parameters = arguments[0]

        v = self.__v
        s = self.__s
        self.__t = 0

        L = len(parameters) // 3

        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            v["dgamma" + str(l)] = np.zeros(parameters["gamma" + str(l)].shape)
            v["dbeta" + str(l)] = np.zeros(parameters["beta" + str(l)].shape)

            s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            s["dgamma" + str(l)] = np.zeros(parameters["gamma" + str(l)].shape)
            s["dbeta" + str(l)] = np.zeros(parameters["beta" + str(l)].shape)

    def optimization(self, *arguments):

        parameters = arguments[0]
        grads = arguments[1]

        v = self.__v
        s = self.__s

        self.__t += 1

        v_corrected = {}
        s_corrected = {}

        L = len(parameters) // 3

        for l in range(1, L + 1):

            v["dW" + str(l)] = self.beta1 * v["dW" + str(l)] + (1 - self.beta1) * grads["dW" + str(l)]
            v["dgamma" + str(l)] = self.beta1 * v["dgamma" + str(l)] + (1 - self.beta1) * grads["dgamma" + str(l)]
            v["dbeta" + str(l)] = self.beta1 * v["dbeta" + str(l)] + (1 - self.beta1) * grads["dbeta" + str(l)]

            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(self.beta1, self.__t))
            v_corrected["dgamma" + str(l)] = v["dgamma" + str(l)] / (1 - np.power(self.beta1, self.__t))
            v_corrected["dbeta" + str(l)] = v["dbeta" + str(l)] / (1 - np.power(self.beta1, self.__t))

            s["dW" + str(l)] = self.beta2 * s["dW" + str(l)] + (1 - self.beta2) * np.power(grads['dW' + str(l)], 2)
            s["dgamma" + str(l)] = self.beta2 * s["dgamma" + str(l)] + (1 - self.beta2) * np.power(
                grads['dgamma' + str(l)], 2)
            s["dbeta" + str(l)] = self.beta2 * s["dbeta" + str(l)] + (1 - self.beta2) * np.power(
                grads['dbeta' + str(l)], 2)

            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(self.beta2, self.__t))
            s_corrected["dgamma" + str(l)] = s["dgamma" + str(l)] / (1 - np.power(self.beta2, self.__t))
            s_corrected["dbeta" + str(l)] = s["dbeta" + str(l)] / (1 - np.power(self.beta2, self.__t))

            parameters["W" + str(l)] = parameters["W" + str(l)]- self.learning_rate * (v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + self.epsilon))
            if v is None:
                break
            if grads is None:
                break
        parameters["gamma" + str(l)] = parameters["gamma" + str(l)]- self.learning_rate * (v_corrected["dgamma" + str(l)] / (np.sqrt(s_corrected["dgamma" + str(l)]) + self.epsilon))
        parameters["beta" + str(l)] = parameters["beta" + str(l)] - self.learning_rate * (v_corrected["dbeta" + str(l)] / (np.sqrt(s_corrected["dbeta" + str(l)]) + self.epsilon))
        return parameters


"""
类型：抽象类
说明：定义决策方法。
"""


class interface_decision(object):
    epsilon = 1e-8

    """
    类型：公有成员变量
    说明：定义决策计算中的激活函数。
    """
    activation = None

    """
    类型：抽象公有成员函数
    说明：定义预测方法。
    """

    def prediction(self, *arguments):
        pass

    """
    类型：抽象公有成员函数
    说明：定义计算精度的方法。
    """

    def accuracy(self, *arguments):
        pass


"""
类型：具体类
说明：定义标准的决策方法。
"""


class decider_standard(interface_decision):

    def prediction(self, *arguments):
        decision_set = arguments[0]
        parameters = arguments[1]

        A = decision_set
        layer_number = len(parameters) // 2

        for l in range(1, layer_number):
            A = self.activation.activate_function(np.dot(parameters["W" + str(l)], A) + parameters["b" + str(l)])

        ZL = np.dot(parameters["W" + str(layer_number)], A) + parameters["b" + str(layer_number)]
        AL = np.exp(ZL) / np.sum(np.exp(ZL), axis=0, keepdims=True)

        return np.argmax(AL, 0)

    def accuracy(self, *arguments):
        decision_set = arguments[0]
        decision_label = arguments[1]
        parameters = arguments[2]

        bool_array = np.equal(self.prediction(decision_set, parameters), np.argmax(decision_label, 0))
        acc = np.sum(bool_array == True) / len(bool_array)

        return acc


"""
类型：具体类
说明：定义带Batch normalization的决策方法。
"""


class decider_BN(interface_decision):

    def prediction(self, *arguments):
        decision_set = arguments[0]
        parameters = arguments[1]

        A = decision_set
        layer_number = len(parameters) // 3

        for l in range(1, layer_number):
            Z = np.dot(parameters["W" + str(l)], A)
            Zmean = np.mean(Z, axis=1, keepdims=True)
            Zvariance = np.mean(np.square(Z - Zmean), axis=1, keepdims=True)
            Ztilde = (Z - Zmean) / np.sqrt(Zvariance + self.epsilon)
            Znorm = np.multiply(parameters["gamma" + str(l)], Ztilde) + parameters["beta" + str(l)]
            A = self.activation.activate_function(Znorm)

        ZL = np.dot(parameters["W" + str(layer_number)], A)
        Zmean = np.mean(ZL, axis=1, keepdims=True)
        Zvariance = np.mean(np.square(ZL - Zmean), axis=1, keepdims=True)
        Ztilde = (ZL - Zmean) / np.sqrt(Zvariance + self.epsilon)
        Znorm = np.multiply(parameters["gamma" + str(layer_number)], Ztilde) + parameters["beta" + str(layer_number)]
        AL = np.exp(Znorm) / np.sum(np.exp(Znorm), axis=0, keepdims=True)

        return np.argmax(AL, 0)

    def accuracy(self, *arguments):
        decision_set = arguments[0]
        decision_label = arguments[1]
        parameters = arguments[2]

        bool_array = np.equal(self.prediction(decision_set, parameters), np.argmax(decision_label, 0))
        acc = np.sum(bool_array == True) / len(bool_array)

        return acc


"""
类型：抽象类
说明：定义学习率衰减方法。
"""


class interface_learning_rate(object):
    """
    类型：公有成员变量
    说明：定义每几步衰减一次。
    """
    epoch_step = 10

    """
    类型：公有成员变量
    说明：衰退比率。
    """
    k = 0.1

    """
    类型：保护成员变量
    说明：迭代次数，用于自加操作。
    """
    _t = 0

    def learning_rate_decay(self, *arguments):
        pass


"""
类型：具体类
说明：指数学习率衰减方法。
"""


class exponential_decay(interface_learning_rate):

    def learning_rate_decay(self, *arguments):
        recent_epoch = arguments[0]
        learning_rate = arguments[1]

        if (recent_epoch % self.epoch_step == 0):
            self._t += 1
            learning_rate = learning_rate * np.exp(-self.k * self._t)

        return learning_rate


"""
类型：具体类
说明：除以t的学习率衰减方法。
"""


class div_t_decay(interface_learning_rate):

    def learning_rate_decay(self, *arguments):
        recent_epoch = arguments[0]
        learning_rate = arguments[1]

        if (recent_epoch % self.epoch_step == 0):
            self._t += 1
            learning_rate = learning_rate / (1 + self.k * self._t)

        return learning_rate


"""
类型：抽象类
说明：定义一个工厂类，用来生产学习模型过程中所需要的模块。
"""


class interface_factory(object):
    """
    类型：公有成员变量
    说明：数据预处理规则化模块。
    """
    normalizer = None

    """
    类型：公有成员变量
    说明：学习参数初始化模块。
    """
    initializer = None

    """
    类型：公有成员变量
    说明：向前向后学习算法模块。
    """
    propagator = None

    """
    类型：公有成员变量
    说明：优化算法模块。
    """
    optimizer = None

    """
    类型：公有成员变量
    说明：决策模块。
    """
    decider = None

    """
    类型：公有成员变量
    说明：学习率衰减模块。
    """
    decay_rater = None

    """
    类型：公有成员变量
    说明：数据批量优化算法说需要的batch大小，即每次训练从训练集里抽样的个数。
    """
    minibatch_size = 512

    """
    类型：公有成员变量
    说明：epoch个数，即需要训练整个数据集多少遍。
    """
    num_epochs = 100

    """
    类型：抽象公有成员函数
    说明：创建一个工厂，该工厂生产训练和测试数据过程中所需要的模块。
    """

    def create_workpiece(self, *arguments):
        pass


"""
类型：具体类
说明：工厂版本v1，不带Batch normalization功能，激活函数为relu，最后一层激活函数为softmax，代价函数用的是交叉熵损失函数，优化算法为adam，学习率为指数衰减。
"""


class factory_v1(interface_factory):
    """
    类型：公有成员函数
    说明：创建一个工厂，该工厂生产训练和测试数据过程中所需要的模块。
    参数：
        arguments[0] -- 数值计算精度 - epsilon
        arguments[1] -- 网络结构 - structure
        arguments[2] -- 随机种子 - seed
        arguments[3] -- 学习率 - learning_rate
        arguments[4] -- adam参数 - beta1
        arguments[5] -- adam参数 - beta2
        arguments[6] -- batch的大小 - minibatch_size
        arguments[7] -- epoch个数 - num_epochs
        arguments[8] -- 衰减步数 - epoch_step
        arguments[9] -- 衰减率 - k
    """

    def create_workpiece(self, *arguments):
        epsilon = arguments[0]

        self.normalizer = zscore_normalization()
        self.normalizer.epsilon = epsilon

        structure = arguments[1]
        seed = arguments[2]

        self.initializer = xavier_initialize_parameters()
        self.initializer.structure = structure
        self.initializer.seed = seed

        self.propagator = propagation_standard()
        self.propagator.epsilon = epsilon
        self.propagator.functor["activation"] = relu()
        self.propagator.functor["activation"].epsilon = epsilon
        self.propagator.functor["cost"] = softmax()
        self.propagator.functor["cost"].epsilon = epsilon

        learning_rate = arguments[3]
        beta1 = arguments[4]
        beta2 = arguments[5]

        self.optimizer = adam_standard()
        self.optimizer.epsilon = epsilon
        self.optimizer.learning_rate = learning_rate
        self.optimizer.beta1 = beta1
        self.optimizer.beta2 = beta2

        self.decider = decider_standard()
        self.decider.epsilon = epsilon
        self.decider.activation = relu()

        self.minibatch_size = arguments[6]
        self.num_epochs = arguments[7]

        epoch_step = arguments[8]
        rate_k = arguments[9]

        self.decay_rater = exponential_decay()
        self.decay_rater.epoch_step = epoch_step
        self.decay_rater.k = rate_k

        return self


"""
类型：具体类
说明：工厂版本v2，带Batch normalization功能，激活函数为relu，最后一层激活函数为softmax，代价函数用的是交叉熵损失函数，优化算法为adam，学习率为指数衰减。
"""


class factory_v2(interface_factory):
    """
    类型：公有成员函数
    说明：创建一个工厂，该工厂生产训练和测试数据过程中所需要的模块。
    参数：
        arguments[0] -- 数值计算精度 - epsilon
        arguments[1] -- 网络结构 - structure
        arguments[2] -- 随机种子 - seed
        arguments[3] -- 学习率 - learning_rate
        arguments[4] -- adam参数 - beta1
        arguments[5] -- adam参数 - beta2
        arguments[6] -- batch的大小 - minibatch_size
        arguments[7] -- epoch个数 - num_epochs
        arguments[8] -- 衰减步数 - epoch_step
        arguments[9] -- 衰减率 - k
    """

    def create_workpiece(self, *arguments):
        epsilon = arguments[0]

        self.normalizer = zscore_normalization()
        self.normalizer.epsilon = epsilon

        structure = arguments[1]
        seed = arguments[2]

        self.initializer = xavier_initialize_parameters_BN()
        self.initializer.structure = structure
        self.initializer.seed = seed

        self.propagator = propagation_BN()
        self.propagator.epsilon = epsilon
        bn = batch_normlizer()
        bn.epsilon = epsilon
        self.propagator.regularizer = {'batch_normlizer': bn}
        self.propagator.functor["activation"] = relu()
        self.propagator.functor["activation"].epsilon = epsilon
        self.propagator.functor["cost"] = softmax()
        self.propagator.functor["cost"].epsilon = epsilon

        learning_rate = arguments[3]
        beta1 = arguments[4]
        beta2 = arguments[5]

        self.optimizer = adam_BN()
        self.optimizer.epsilon = epsilon
        self.optimizer.learning_rate = learning_rate
        self.optimizer.beta1 = beta1
        self.optimizer.beta2 = beta2

        self.decider = decider_BN()
        self.decider.epsilon = epsilon
        self.decider.activation = relu()

        self.minibatch_size = arguments[6]
        self.num_epochs = arguments[7]

        epoch_step = arguments[8]
        rate_k = arguments[9]

        self.decay_rater = exponential_decay()
        self.decay_rater.epoch_step = epoch_step
        self.decay_rater.k = rate_k

        return self


"""
类型：具体类
说明：工厂版本v3，带Batch normalization功能，带dropout规则化，激活函数为relu，最后一层激活函数为softmax，代价函数用的是交叉熵损失函数，
      优化算法为adam，学习率为指数衰减。
"""


class factory_v3(interface_factory):
    """
    类型：公有成员函数
    说明5：创建一个工厂，该工厂生产训练和测试数据过程中所需要的模块。
    参数：
        arguments[0] -- 数值计算精度 - epsilon
        arguments[1] -- 网络结构 - structure
        arguments[2] -- 随机种子 - seed
        arguments[3] -- 学习率 - learning_rate
        arguments[4] -- adam参数 - beta1
        arguments[5] -- adam参数 - beta2
        arguments[6] -- batch的大小 - minibatch_size
        arguments[7] -- epoch个数 - num_epochs
        arguments[8] -- 衰减步数 - epoch_step
        arguments[9] -- 衰减率 - k
        arguments[10] -- dropout保持概率 - keep_prob
    """

    def create_workpiece(self, *arguments):
        epsilon = arguments[0]

        self.normalizer = zscore_normalization()
        self.normalizer.epsilon = epsilon

        structure = arguments[1]
        seed = arguments[2]

        self.initializer = xavier_initialize_parameters_BN()
        self.initializer.structure = structure
        self.initializer.seed = seed

        self.propagator = propagation_BN()
        self.propagator.epsilon = epsilon
        bn = batch_normlizer()
        bn.epsilon = epsilon
        self.propagator.regularizer['batch_normlizer'] = bn
        self.propagator.functor["activation"] = relu()
        self.propagator.functor["activation"].epsilon = epsilon
        self.propagator.functor["cost"] = softmax()
        self.propagator.functor["cost"].epsilon = epsilon

        learning_rate = arguments[3]
        beta1 = arguments[4]
        beta2 = arguments[5]

        self.optimizer = adam_BN()
        self.optimizer.epsilon = epsilon
        self.optimizer.learning_rate = learning_rate
        self.optimizer.beta1 = beta1
        self.optimizer.beta2 = beta2

        self.decider = decider_BN()
        self.decider.epsilon = epsilon
        self.decider.activation = relu()

        self.minibatch_size = arguments[6]
        self.num_epochs = arguments[7]

        epoch_step = arguments[8]
        rate_k = arguments[9]

        self.decay_rater = exponential_decay()
        self.decay_rater.epoch_step = epoch_step
        self.decay_rater.k = rate_k

        d = dropout()
        d.epsilon = epsilon
        d.keep_prob = arguments[10]

        self.propagator.regularizer['dropout'] = d

        return self


"""
类型：抽象类
说明：定义一个工厂类，用来生产学习模型过程中所需要的模块。
"""


class interface_train_model(object):
    """
    类型：公有成员变量
    说明：学习参数。
    """
    parameters = None

    """
    类型：公有成员变量
    说明：训练集。
    """
    training_set = None

    """
    类型：公有成员变量
    说明：训练标签。
    """
    training_label = None

    """
    类型：公有成员变量
    说明：验证集。
    """
    validation_set = None

    """
    类型：公有成员变量
    说明：验证标签。
    """
    validation_label = None

    """
    类型：公有成员变量
    说明：测试集。
    """
    test_set = None

    """
    类型：公有成员变量
    说明：测试标签。
    """
    test_label = None

    """
    类型：公有成员变量
    说明：工厂对象。
    """
    factory = None

    """
    类型：抽象公有成员函数
    说明：预测方法。
    """

    def prediction(self, *arguments):
        pass

    """
    类型：抽象公有成员函数
    说明：计算训练集精度。
    """

    def training_accuracy(self, *arguments):
        pass

    """
    类型：抽象公有成员函数
    说明：计算验证集精度。
    """

    def validation_accuracy(self, *arguments):
        pass

    """
    类型：抽象公有成员函数
    说明：计算测试集精度。
    """

    def test_accuracy(self, *arguments):
        pass

    """
    类型：抽象公有成员函数
    说明：训练模型方法。
    """

    def training_model(self, *arguments):
        pass

    #def training_cifar_model(self, *arguments,num):
        #pass


"""
类型：具体类
说明：定义深度神经网络。
"""


class deep_neural_networks(interface_train_model):
    """
    类型：初始化函数
    说明：初始化工厂对象。
    """

    def __init__(self, factory):
        self.factory = factory

    def prediction(self, *arguments):

        return self.factory.decider.prediction(self.test_set, self.parameters)

    def training_accuracy(self, *arguments):

        return self.factory.decider.accuracy(self.training_set, self.training_label, self.parameters)

    def validation_accuracy(self, *arguments):

        return self.factory.decider.accuracy(self.validation_set, self.validation_label, self.parameters)

    def test_accuracy(self, *arguments):

        return self.factory.decider.accuracy(self.test_set, self.test_label, self.parameters)

    def training_model(self, *arguments):

        # 定义深度神经网络结构
        self.factory.initializer.structure = np.append(np.array([self.training_set.shape[0]]),
                                                       self.factory.initializer.structure)
        # 定义一轮epoch需要多少batch
        num_minibatches = int(self.training_set.shape[1] / self.factory.minibatch_size) + 1

        # 定义数据预处理步骤
        if (self.factory.normalizer != None):
            self.training_set = self.factory.normalizer.normalize_data(self.training_set)
            self.validation_set = self.factory.normalizer.normalize_data(self.validation_set)

        # 定义学习参数及超参数初始化步骤
        self.parameters = self.factory.initializer.initialize_parameters()
        seed = self.factory.initializer.seed
        self.factory.optimizer.initialization(self.parameters)

        costs = []
        training_accuracies = []
        validation_accuracies = []

        # 在验证集上表现最好的学习参数
        best_parameters = {}
        # 最好学习参数时，训练集的精度
        best_training_acc = 0.
        # 最好学习参数时，验证集的精度
        best_validation_acc = 0.

        start = time.clock()

        # epoch学习过程
        for iter_epoch in range(1, self.factory.num_epochs + 1):
            cost = 0
            # minibatch学习过程
            for iter_batch in range(1, num_minibatches + 1):
                seed += 1
                (minibatch_X, minibatch_Y) = self.__random_mini_batches(self.factory.minibatch_size, seed)
                cost += self.factory.propagator.forward_propagation(minibatch_X, minibatch_Y, self.parameters)
                grads = self.factory.propagator.backward_propagation(minibatch_X, minibatch_Y, self.parameters)
                self.factory.optimizer.optimization(self.parameters, grads)

            if (self.factory.decay_rater != None):
                self.factory.optimizer.learning_rate = self.factory.decay_rater.learning_rate_decay(iter_epoch,
                                                                                                    self.factory.optimizer.learning_rate)

            cost /= num_minibatches
            costs.append(cost)
            training_acc = self.training_accuracy()
            training_accuracies.append(training_acc)
            validation_acc = self.validation_accuracy()
            validation_accuracies.append(validation_acc)

            if (validation_acc > best_validation_acc):
                best_training_acc = training_acc
                best_validation_acc = validation_acc
                best_parameters = copy.deepcopy(self.parameters)

            print("Cost after epoch %i: %f" % (iter_epoch, cost))
            print("Training accuracy after epoch %i: %f" % (iter_epoch, training_acc))
            print("Validation accuracy after epoch %i: %f" % (iter_epoch, validation_acc))

        end = time.clock()

        self.factory.initializer.seed = seed
        self.parameters = best_parameters

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.title("Learning rate =" + str(self.factory.optimizer.learning_rate))
        plt.show()

        plt.plot(np.squeeze(training_accuracies))
        plt.ylabel('training accracy')
        plt.xlabel('epochs')
        plt.title("Learning rate =" + str(self.factory.optimizer.learning_rate))
        plt.show()

        plt.plot(np.squeeze(validation_accuracies))
        plt.ylabel('validation accracy')
        plt.xlabel('epochs')
        plt.title("Learning rate =" + str(self.factory.optimizer.learning_rate))
        plt.show()

        print("Accuracy of total Training set: %f%%" % (best_training_acc * 100))
        print("Accuracy of total Validation set: %f%%" % (best_validation_acc * 100))
        print("Training of total Time: %f Minutes" % ((end - start) / 60))

    def cifar_init(self):
        # 定义一轮epoch需要多少batch
        num_minibatches = int(self.training_set.shape[1] / self.factory.minibatch_size) + 1
        seed = self.factory.initializer.seed
        # 初始化网络
        # 定义深度神经网络结构
        self.factory.initializer.structure = np.append(np.array([self.training_set.shape[0]]),
                                                           self.factory.initializer.structure)
        # 定义数据预处理步骤
        if (self.factory.normalizer != None):
            self.training_set = self.factory.normalizer.normalize_data(self.training_set)
            self.validation_set = self.factory.normalizer.normalize_data(self.validation_set)

        # 定义学习参数及超参数初始化步骤
        self.parameters = self.factory.initializer.initialize_parameters()
        self.factory.optimizer.initialization(self.parameters)
        costs = []
        training_accuracies = []
        validation_accuracies = []
        # 在验证集上表现最好的学习参数
        best_parameters = {}
        # 最好学习参数时，训练集的精度
        best_training_acc = 0.
        # 最好学习参数时，验证集的精度
        best_validation_acc = 0.

        return num_minibatches,seed,costs,training_accuracies,validation_accuracies,best_parameters,best_training_acc,best_validation_acc

    def training_cifar_model(self,num,num_minibatches,seed,costs,training_accuracies,validation_accuracies,best_parameters,best_training_acc,best_validation_acc):
        '''
        # 定义一轮epoch需要多少batch
        num_minibatches = int(self.training_set.shape[1] / self.factory.minibatch_size) + 1
        seed = self.factory.initializer.seed
        # 第一批次进入时候初始化网络
        if num==0:
            # 定义深度神经网络结构
            self.factory.initializer.structure = np.append(np.array([self.training_set.shape[0]]),
                                                           self.factory.initializer.structure)
            # 定义数据预处理步骤
            if (self.factory.normalizer != None):
                self.training_set = self.factory.normalizer.normalize_data(self.training_set)
                self.validation_set = self.factory.normalizer.normalize_data(self.validation_set)

            # 定义学习参数及超参数初始化步骤
            self.parameters = self.factory.initializer.initialize_parameters()
            self.factory.optimizer.initialization(self.parameters)
        costs = []
        training_accuracies = []
        validation_accuracies = []
        # 在验证集上表现最好的学习参数
        best_parameters = {}
        # 最好学习参数时，训练集的精度
        best_training_acc = 0.
        # 最好学习参数时，验证集的精度
        best_validation_acc = 0.
        #start = time.clock()
        '''
        # epoch学习过程
        for iter_epoch in range(1, self.factory.num_epochs + 1):
            cost = 0
            # minibatch学习过程
            for iter_batch in range(1, num_minibatches + 1):
                seed += 1
                (minibatch_X, minibatch_Y) = self.__random_mini_batches(self.factory.minibatch_size, seed)
                cost += self.factory.propagator.forward_propagation(minibatch_X, minibatch_Y, self.parameters)
                grads = self.factory.propagator.backward_propagation(minibatch_X, minibatch_Y, self.parameters)
                self.factory.optimizer.optimization(self.parameters, grads)

            if (self.factory.decay_rater != None):
                self.factory.optimizer.learning_rate = self.factory.decay_rater.learning_rate_decay(iter_epoch,
                                                                                                    self.factory.optimizer.learning_rate)

            cost /= num_minibatches
            costs.append(cost)
            training_acc = self.training_accuracy()
            training_accuracies.append(training_acc)
            validation_acc = self.validation_accuracy()
            validation_accuracies.append(validation_acc)

            if (validation_acc > best_validation_acc):
                best_training_acc = training_acc
                best_validation_acc = validation_acc
                best_parameters = copy.deepcopy(self.parameters)

            print("Cost after epoch %i: %f" % (iter_epoch, cost))
            print("Training accuracy after epoch %i: %f" % (iter_epoch, training_acc))
            print("Validation accuracy after epoch %i: %f" % (iter_epoch, validation_acc))

        return costs, training_accuracies, validation_accuracies, best_training_acc, best_validation_acc
    '''
        if num==4:
            #end = time.clock()

            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('epochs')
            plt.title("Learning rate =" + str(self.factory.optimizer.learning_rate))
            plt.show()

            plt.plot(np.squeeze(training_accuracies))
            plt.ylabel('training accracy')
            plt.xlabel('epochs')
            plt.title("Learning rate =" + str(self.factory.optimizer.learning_rate))
            plt.show()

            plt.plot(np.squeeze(validation_accuracies))
            plt.ylabel('validation accracy')
            plt.xlabel('epochs')
            plt.title("Learning rate =" + str(self.factory.optimizer.learning_rate))
            plt.show()

            print("Accuracy of total Training set: %f%%" % (best_training_acc * 100))
            print("Accuracy of total Validation set: %f%%" % (best_validation_acc * 100))
            #print("Training of total Time: %f Minutes" % ((end - start) / 60))
    '''
    def cifar_plot(self,costs, training_accuracies, validation_accuracies, best_training_acc, best_validation_acc):
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.title("Learning rate =" + str(self.factory.optimizer.learning_rate))
        plt.show()

        plt.plot(np.squeeze(training_accuracies))
        plt.ylabel('training accracy')
        plt.xlabel('epochs')
        plt.title("Learning rate =" + str(self.factory.optimizer.learning_rate))
        plt.show()

        plt.plot(np.squeeze(validation_accuracies))
        plt.ylabel('validation accracy')
        plt.xlabel('epochs')
        plt.title("Learning rate =" + str(self.factory.optimizer.learning_rate))
        plt.show()
        print("Accuracy of total Training set: %f%%" % (best_training_acc * 100))
        print("Accuracy of total Validation set: %f%%" % (best_validation_acc * 100))

    """
    类型：私有成员函数
    说明：从训练数据集和标签集中随机抽取minibatch。
    参数：
        X_train -- 训练数据集。
        Y_train -- 训练标签集。
        minibatch_size -- minibatch大小。
        seed -- 随机种子。
    返回值：
        元组 -- 返回一个minibatch，包括训练数据和对应的标签数据。
    """

    def __random_mini_batches(self, minibatch_size, seed):

        np.random.seed(seed)
        shuffle_array = np.random.randint(0, self.training_set.shape[1], minibatch_size)
        return (self.training_set[:, shuffle_array], self.training_label[:, shuffle_array])


def main():
    print("开始读取MNIST数据")
    path = 'dataset/MNIST'
    train_images, train_labels = load_mnist.load_mnist(path, kind='train')  # train_images 60000*784 train_lable 1*60000
    train_labels=One_Hot_Encode_with_Keras.get_one_hot(train_labels)
    test_images,test_labels=load_mnist.load_mnist(path,kind='t10k') #test_images 10000*784 test_lable 1*10000
    test_labels = One_Hot_Encode_with_Keras.get_one_hot(test_labels)
    print("MNIST数据读取完成")

    print("开始模型参数初始化")
    struct = [512, 256, 128, 64, 32, 16, 10]
    factory = factory_v2()
    factory.create_workpiece(1e-8, struct, 1, 1e-3, 0.9, 0.999, 512, 200, 5, 0.01)

    dnn = deep_neural_networks(factory)
    print("模型参数初始化完成")

    print("加密开始")
    #w=16
    encrypt_start_time=time.time()
    #e=EIVHE.EIVHE(train_images,test_images,w)
    train_images = np.loadtxt(open("encrypted_train_images.csv","rb"), delimiter=",", skiprows=0)
    train_images=np.abs(train_images)#取正数
    #train_images,sk,test_images,sk2=e.EIVHE_encrypt()
    encrypt_end_time=time.time()
    use_time=encrypt_end_time-encrypt_start_time
    print("数据加密完成,用时%f秒"%use_time)
    #np.savetxt('encrypted_train_images.csv',train_images,delimiter=',')
    #np.savetxt('encrypted_test_images.csv', test_images, delimiter=',')
    #np.savetxt('secret_key.csv', sk, delimiter=',')
    print("数据保存完毕")
    print("训练开始：")
    dnn.training_set = train_images.T
    dnn.training_label = train_labels.T
    test_images=np.loadtxt(open('encrypted_test_images.csv','rb'),delimiter=',',skiprows=0)
    test_images=np.abs(test_images)
    dnn.validation_set = test_images.T  #lable采用one-hot编码
    dnn.validation_label = test_labels.T
    dnn.training_model()


    '''
    '''
    #MNIST原始
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    struct = [512, 256, 128, 64, 32, 16, 10]
    factory = factory_v2()
    factory.create_workpiece(1e-8, struct, 1, 1e-3, 0.9, 0.999, 512, 100, 5, 0.01)

    dnn = deep_neural_networks(factory)
    dnn.training_set = mnist.train.images.T
    dnn.training_label = mnist.train.labels.T
    dnn.validation_set = mnist.test.images.T
    dnn.validation_label = mnist.test.labels.T
    dnn.training_model()
    '''


def CIFAR_main():
    #读取CIFAR数据
    #训练集

    x = 5
    #CIFAR_data = np.ones((10000, 3072)).astype('int')
    #CIFAR_label = np.ones(10000).astype('int')
    #测试集
    #CIFAR_test_data=np.ones((10000,3072)).astype('int')
    #CIFAR_test_label=np.ones(10000).astype('int')
    CIFAR_test_data=np.array(load_CIFAR10.load_file('dataset/CIFAR-10/test_batch').get("data")).astype('int')
    CIFAR_test_label=One_Hot_Encode_with_Keras.get_one_hot( np.array(load_CIFAR10.load_file('dataset/CIFAR-10/test_batch').get("labels")) ).astype('int')

    print("开始模型参数初始化")
    init_start=time.time()
    struct = [512, 256, 128, 64, 32, 16, 10]
    factory = factory_v2()
    factory.create_workpiece(1e-8, struct, 1, 1e-3, 0.9, 0.999, 512, 200, 5, 0.01)

    dnn = deep_neural_networks(factory)
    tem_data = np.loadtxt(open("encrypted_train_images_0.csv", "rb"), delimiter=",", skiprows=0)#临时存储
    tem_test = np.loadtxt(open('encrypted_test_images_0.csv', 'rb'), delimiter=',', skiprows=0) #临时存储
    dnn.training_set = tem_data.T
    dnn.validation_set = tem_test.T
    num_minibatches,seed,costs,training_accuracies,validation_accuracies,best_parameters,best_training_acc,best_validation_acc = dnn.cifar_init()#初始化参数
    init_end=time.time()
    print("模型参数初始化完成,用时 %f 秒" % (init_end-init_start))
    start = time.clock()#计算总的计算时间
    for i in range(0,x):
        print("开始读取第"+str(i+1)+"批次CIFAR数据：")
        CIFAR_data = np.array(
            load_CIFAR10.load_file('dataset/CIFAR-10/data_batch_' + str(i + 1)).get("data"))
        CIFAR_label = np.array(
            load_CIFAR10.load_file('dataset/CIFAR-10/data_batch_' + str(i + 1)).get("labels"))
        CIFAR_label = One_Hot_Encode_with_Keras.get_one_hot(CIFAR_label)
        print("第"+str(i+1)+"批次CIFAR数据读取完成")

        print("加密开始")
        #w = 16
        encrypt_start_time = time.time()
        #e = EIVHE.EIVHE(CIFAR_data, CIFAR_test_data, w)
        CIFAR_data = np.loadtxt(open("encrypted_train_images_"+str(i+1)+".csv", "rb"), delimiter=",", skiprows=0) #读取加密的训练数据集
        CIFAR_test_data = np.loadtxt(open('encrypted_test_images_'+str(i+1)+'.csv', 'rb'), delimiter=',', skiprows=0) #读取加密的测试数据集
        # train_images=np.abs(train_images)#取正数
        #CIFAR_data, sk, CIFAR_test_data, sk2 = e.EIVHE_encrypt()
        encrypt_end_time = time.time()
        use_time = encrypt_end_time - encrypt_start_time
        print("数据加密完成,用时%f秒" % use_time)
        #np.savetxt('encrypted_train_images_'+str(i+1)+'.csv', CIFAR_data, delimiter=',')
        #np.savetxt('encrypted_test_images_'+str(i+1)+'.csv', CIFAR_test_data, delimiter=',')
        #np.savetxt('secret_key_'+str(i+1)+'.csv', sk, delimiter=',')
        #print("数据保存完毕")

        print("训练开始：")
        dnn.training_set = CIFAR_data.T
        dnn.training_label = CIFAR_label.T
        # test_images=np.abs(test_images)
        dnn.validation_set = CIFAR_test_data.T  # lable采用one-hot编码
        dnn.validation_label = CIFAR_test_label.T
        batch_time_start=time.clock()#计算每个batch计算的时间
        #dnn.training_cifar_model(i)
        costs, training_accuracies, validation_accuracies, best_training_acc, best_validation_acc = dnn.training_cifar_model(i, num_minibatches, seed, costs, training_accuracies, validation_accuracies, best_parameters, best_training_acc, best_validation_acc)
        batch_time_end=time.clock()
        print("Training of one batch Time: %f Minutes" % ((batch_time_end - batch_time_start) / 60))
    dnn.cifar_plot(costs, training_accuracies, validation_accuracies, best_training_acc, best_validation_acc)
    end = time.clock()
    print("Training of total Time: %f Minutes" % ((end - start) / 60))

def CIFAR_ori_main():
    CIFAR_data = np.ones((50000, 3072)).astype('int')
    CIFAR_label = np.ones(50000).astype('int')
    for i in range(5):
        print("开始读取第" + str(i + 1) + "批次CIFAR数据：")
        CIFAR_data[i * 10000:(i + 1) * 10000, :] = np.array(
            load_CIFAR10.load_file('dataset/CIFAR-10/data_batch_' + str(i + 1)).get("data"))
        CIFAR_label[i * 10000: (i + 1) * 10000] = np.array(
            load_CIFAR10.load_file('dataset/CIFAR-10/data_batch_' + str(i + 1)).get("labels"))
        print("第" + str(i + 1) + "批次CIFAR数据读取完成")
    CIFAR_label = One_Hot_Encode_with_Keras.get_one_hot(CIFAR_label)

    CIFAR_test_data = np.array(load_CIFAR10.load_file('dataset/CIFAR-10/test_batch').get("data")).astype('int')
    CIFAR_test_label = One_Hot_Encode_with_Keras.get_one_hot(np.array(load_CIFAR10.load_file('dataset/CIFAR-10/test_batch').get("labels")).astype('int') )

    struct = [3072, 2048, 1024, 512, 256, 128, 64, 32, 16, 10]
    factory = factory_v2()
    factory.create_workpiece(1e-8, struct, 1, 1e-3, 0.9, 0.999, 2048, 200, 10, 0.01)

    dnn = deep_neural_networks(factory)
    dnn.training_set = CIFAR_data.T
    dnn.training_label = CIFAR_label.T
    dnn.validation_set = CIFAR_test_data.T
    dnn.validation_label = CIFAR_test_label.T
    dnn.training_model()

#CIFATA加密训练神经网络测试函数
def CIFAR_en_main():
    CIFAR_data = np.ones((50000, 3073)).astype('int')
    CIFAR_label = np.ones(50000).astype('int')
    #训练数据集
    for i in range(5):
        print("开始读取第"+str(i+1)+"批次CIFAR数据：")
        CIFAR_data[i * 10000:(i + 1) * 10000, :] = np.array(
            np.loadtxt(open("encrypted_train_images_" + str(i + 1) + ".csv", "rb"), delimiter=",", skiprows=0))
        CIFAR_label[i * 10000 : (i + 1) * 10000] = np.array(
            load_CIFAR10.load_file('dataset/CIFAR-10/data_batch_' + str(i + 1)).get("labels"))
        print("第"+str(i+1)+"批次CIFAR数据读取完成")

    CIFAR_label = One_Hot_Encode_with_Keras.get_one_hot(CIFAR_label)
    #测试数据集

    CIFAR_test_data = np.array(np.loadtxt(open("encrypted_test_images_0.csv", "rb"), delimiter=",", skiprows=0)).astype('int')
    CIFAR_test_label = One_Hot_Encode_with_Keras.get_one_hot(
        np.array(load_CIFAR10.load_file('dataset/CIFAR-10/test_batch').get("labels")).astype('int'))


    CIFAR_data=np.abs(CIFAR_data)
    CIFAR_test_data=np.abs(CIFAR_test_data)
    #神经网路的训练
    struct = [3073, 2048, 1024, 512, 256, 128, 64, 32, 16, 10]
    factory = factory_v2()
    factory.create_workpiece(1e-8, struct, 1, 1e-3, 0.9, 0.999, 2048, 5, 5, 0.01)

    dnn = deep_neural_networks(factory)
    dnn.training_set = CIFAR_data.T
    dnn.training_label = CIFAR_label.T
    dnn.validation_set = CIFAR_test_data.T
    dnn.validation_label = CIFAR_test_label.T
    dnn.training_model()

def fashion_mnist():
    print("开始读取FATION_MNIST数据")
    path = 'dataset/FASHION_MNIST'
    train_images, train_labels = load_mnist.load_mnist(path, kind='train')  # train_images 60000*784 train_lable 1*60000
    rain_labels = One_Hot_Encode_with_Keras.get_one_hot(train_labels)
    test_images, test_labels = load_mnist.load_mnist(path, kind='t10k')  # test_images 10000*784 test_lable 1*10000
    test_labels = One_Hot_Encode_with_Keras.get_one_hot(test_labels)
    print("FATION_MNIST数据读取完成")

    print("开始模型参数初始化")
    struct = [512, 256, 128, 64, 32, 16, 10]
    factory = factory_v2()
    factory.create_workpiece(1e-8, struct, 1, 1e-2, 0.9, 0.999, 512, 100, 10, 0.01)
    dnn = deep_neural_networks(factory)
    print("模型参数初始化完成")

    print("加密开始")
    #w=16
    encrypt_start_time = time.time()
    #e=EIVHE.EIVHE(train_images,test_images,w)
    train_images = np.loadtxt(open("encrypted_fm_train_images.csv", "rb"), delimiter=",", skiprows=0)
    train_images=np.abs(train_images)#取正数
    #train_images,sk,test_images,sk2=e.EIVHE_encrypt()
    encrypt_end_time = time.time()
    use_time = encrypt_end_time - encrypt_start_time
    print("数据加密完成,用时%f秒" % use_time)
    #np.savetxt('encrypted_fm_train_images.csv',train_images,delimiter=',')
    #np.savetxt('encrypted_fm_test_images.csv', test_images, delimiter=',')
    #np.savetxt('secret_fm_key.csv', sk, delimiter=',')
    print("数据保存完毕")
    print("训练开始：")
    dnn.training_set = train_images.T
    dnn.training_label = train_labels.T
    test_images = np.loadtxt(open('encrypted_fm_test_images.csv', 'rb'), delimiter=',', skiprows=0)
    test_images=np.abs(test_images)
    dnn.validation_set = test_images.T  # lable采用one-hot编码
    dnn.validation_label = test_labels.T
    dnn.training_model()

def fashion_mnist_ori():
    print("开始读取FATION_MNIST数据")
    path = 'dataset/FASHION_MNIST'
    train_images, train_labels = load_mnist.load_mnist(path, kind='train')  # train_images 60000*784 train_lable 1*60000
    train_labels = One_Hot_Encode_with_Keras.get_one_hot(train_labels)
    test_images, test_labels = load_mnist.load_mnist(path, kind='t10k')  # test_images 10000*784 test_lable 1*10000
    test_labels = One_Hot_Encode_with_Keras.get_one_hot(test_labels)
    print("FATION_MNIST数据读取完成")

    print("开始模型参数初始化")
    struct = [512, 256, 128, 64, 32, 16, 10]
    factory = factory_v2()
    factory.create_workpiece(1e-8, struct, 1, 1e-2, 0.9, 0.999, 512, 100, 10, 0.01)
    dnn = deep_neural_networks(factory)
    print("模型参数初始化完成")
    print("训练开始：")
    dnn.training_set = train_images.T
    dnn.training_label = train_labels.T
    test_images=np.abs(test_images)
    dnn.validation_set = test_images.T  # lable采用one-hot编码
    dnn.validation_label = test_labels.T
    dnn.training_model()

if __name__=="__main__":
    #CIFAR_main()
    #CIFAR_ori_main()
    #CIFAR_en_main()
    #fashion_mnist()
    #fashion_mnist_ori()
    main()
