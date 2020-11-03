import numpy as np

class EIVHE(object):
    def __init__(self,x,y,w):#初始化
        self.x=x
        self.w=w
        self.y=y

    def __generate_key(self,w,m,n):
        S = (np.random.rand(m,n) * w / (2 ** 26)) # 可证明 max(S) < w
        S = np.round(S*1000)
        return S

    def __encrypt(self,x,S,m,n,w):
        assert x.shape[1] == S.shape[1]

        e = (np.random.rand(m,n)) # 可证明 max(e) < w / 2
        c = ((w * x) + e).dot(np.linalg.inv(S))
        return c

    def __decrypt(self,c,S,w):
        #return (S.dot(c) / w).astype('int')
        return (np.dot(c,S.T)/w).astype('int')

    def __switch_key(self,c,S,m,n,T):
        l = int(np.ceil(np.log2(np.max(np.abs(c)))))
        c_star = self.__get_c_star(c,m,l)
        S_star = self.__get_S_star(S,m,n,l)
        n_prime = n + 1

        S_prime = np.concatenate((np.eye(n),T.T),0).T
        A = (np.random.rand(n_prime - n, n*l) * 10).astype('int')
        E = (1 * np.random.rand(S_star.shape[0],S_star.shape[1])).astype('int')
        M = np.concatenate(((S_star - T.dot(A) + E),A),0)
        #c_prime = M.dot(c_star)
        c_prime=np.dot(c_star,M.T)
        return c_prime,S_prime

    def __get_c_star(self,c,m,l):
        n= c.shape[1]#n求出密文的列数
        c_star = np.zeros((m,l * n),dtype='int')#l*n表示一行的代码用多少位来存储
        for j in range(m):
            for i in range(n):
                b = np.array(list(np.binary_repr(np.abs(c[j][i]))), dtype='int')#b表示一个数字的二进制编码
                if(c[j][i] < 0):
                    b *= -1
                c_star[j][(i * l) + (l-len(b)) : (i+1) * l] += b
        #c_star=c_star.reshape((m,l * n))
        assert c_star.shape[0]==m
        assert c_star.shape[1]==n*l
        return c_star

    def __get_S_star(self,S,m,n,l):
        #S_star = np.ones((MNIST,l*m),dtype='int')
        S_star = list()
        for i in range(l):
            S_star.append(S*2**(l-i-1))
        S_star = np.array(S_star).transpose(1,2,0).reshape(n,l*n)
        return S_star

    def __get_T(self,n):
        n_prime = n + 1
        T = (10 * np.random.rand(n,n_prime-n)).astype('int')
        return T

    def __encrypt_via_switch(self,x,w,m,n,T,S):
        #c=encrypt(x,S,m,n,w)
        #c=np.floor( np.dot(w*x,np.linalg.inv(S))).astype('int')#向下取整
        c= w * x
        S=np.eye(n)
        c,S = self.__switch_key(c,S,m,n,T)
        return c,S

    def EIVHE_encrypt(self):

        m=self.x.shape[0]#行
        n=self.x.shape[1]#列

        S = self.__generate_key(self.w,n,n) #密钥的生成
        T= self.__get_T(n)
        c1,S1=self.__encrypt_via_switch(self.x,self.w,m,n,T,S)
        h=self.y.shape[0]
        c2,S2=self.__encrypt_via_switch(self.y,self.w,h,n,T,S)
        return c1,S1,c2,S2
        #return c1,S1


    def EIVHE_decrypt(self,c,S,w):
        return self.__decrypt(c,S,w)
