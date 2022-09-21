import math
import tenseal as ts
from time import time
import numpy as np
import multiprocessing

def test(x, i, dic):
    dic[i] = x.pow(2)

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    lst = multiprocessing.Manager().dict()
    start = time()
    for i in range(1, 10):
        pool.apply_async(test, (i, i, lst))
    pool.close()
    pool.join()
    end = time()
    print(f'Time:{end - start}')
