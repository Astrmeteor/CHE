#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import time
from tqdm import tqdm
import os
import math 
import pathos
from itertools import product
from seal import *
import multiprocessing as mp
import pickle
import concurrent.futures


# In[ ]:


def weight_change(weight, parms):
    """
    :param weight: weight of conv layer
    :param parms: encryption parameter dictionary
    :return: changed weight
    """
    weight = weight.numpy()
    weight_temp = np.zeros(weight.shape).tolist()
    for outer, inner, i in product(range(weight.shape[0]), range(weight.shape[1]), range(weight.shape[2])):
        weight_temp[outer][inner][i] = parms['encoder'].encode(weight[outer][inner][i].item(), parms['scale'])
        weight_temp[outer][inner][i].set_parms(parms["parms"])
    return weight_temp


# In[ ]:


def bias_change(bias, parms):
    """
    :param bias: bias of conv layer
    :param parms: encryption parameter dictionary
    :return:
    """
    for i in range(bias.__len__()):
        bias[i] = parms['encoder'].encode(bias[i], parms['scale'])
        bias[i].set_parms(parms["parms"])
    return bias


# In[ ]:


def coefficients(mu, sigma, gamma, beta, eps, name):
    """
    :param mu: BN mu
    :param sigma: BN sigma
    :param gamma: BN gamma
    :param beta: BN beta
    :param eps: BN eps
    :param name: BN paradigm, CBA or CAB
    :return: coefficients merging
    """
    if name == 'CBA':
        temp = gamma / np.sqrt(sigma + eps)
        a = np.power(temp, 2)
        b = 2 * (beta - temp * mu) * temp
        c = np.power((beta - temp * mu), 2)
    elif name == 'CAB':
        a = gamma / np.sqrt(sigma + eps)
        b = 0
        c = beta - a * mu
    return a, b / a, c / a


# In[ ]:


def weight_diag(weight):
    """
    :param weight: weight of FC layer
    :return: padded and diagonalization weight of FC layer
    """
    weight = np.array(weight)
    weight = np.pad(weight, pad_width=((0, 0), (0, weight.shape[0] - weight.shape[1])))
    W = []
    for i in range(weight.shape[0]):
        if i == 0:
            W.append(np.diag(weight, i).tolist())
        else:
            W.append(np.concatenate([np.diag(weight, -i), np.diag(weight, weight.shape[0] - i)]).tolist())
    return W


# In[ ]:


def count_rotation(m, f):
    """
    :param m: input size
    :param f: filter size
    :param p: padding number
    :return: rotation index
    """
    idx = []
    for i in range(f):
        start = i * m
        for j in range(f):
            if i == f - 1 & j == f - 1:
                continue
            a = start + j
            if j == f - 1:
                a = (i + 1) * m - 1
                idx.append(a + 1)
            else:
                idx.append(a + 1)
    assert len(idx) == pow(f, 2) - 1
    return idx


# In[ ]:


def multi_conv_rot(input, j, idx, gk, parms, lst):
    context = SEALContext(parms)
    eva = Evaluator(context)
    x = eva.rotate_vector(input, idx[j - 1], gk)
    x.set_parms(parms)
    lst.append(x)


# In[ ]:


def multi_conv(conv_weight, conv_bias, inputs, f, mask, gk, parms, lst):
    """
    multiprocessing
    :param conv_weight: weight of each output channel
    :param conv_bias: bias of each output channel
    :param inputs: input feature maps (fixed)
    :param f: filter size (fixed)
    :param mask: mask (fixed)
    :param parms: encryption parameters (fixed)
    :return: result of each output channel: lst
    """
    # compute element-wise and accumulation
    sss = time.time()
    in_channel = conv_weight.__len__()
    Z = []
    context = SEALContext(parms)
    eva = Evaluator(context)
    for i in range(in_channel):
        Y = []
        for j in range(f * f):
            eva.mod_switch_to_inplace(conv_weight[i][j], inputs[i][j].parms_id())
            Y.append(eva.multiply_plain(inputs[i][j], conv_weight[i][j]))
        Z.append(eva.add_many(Y))
    Z = eva.add_many(Z)
    eva.rescale_to_next_inplace(Z)

    # add bias
    eva.mod_switch_to_inplace(conv_bias, Z.parms_id())
    Z.scale(2 ** int(math.log2(conv_bias.scale())))
    eva.add_plain_inplace(Z, conv_bias)

    eva.mod_switch_to_inplace(mask, Z.parms_id())
    eva.multiply_plain_inplace(Z, mask)
    eva.rescale_to_next_inplace(Z)
    
    Z.set_parms(parms)
    print(time.time() - sss)
    #lst.append(pickle.dumps(Z))
    lst.append(Z)
    # return Z


# In[ ]:


"""
# Inputs Rotation
# inputs = [[0 for i in range(f * f)] for j in range(in_channel)]
rot_workers = []
rot_dic = {}
for i in range(input.__len__()):
    rot_dic['rot_lst'+ str(i)] = mp.Manager().list()
    # rot_lst = mp.Manager().list()
    input[i].set_parms(parms["parms"])
    for j in range(f*f):
        p = mp.Process(target=multi_conv_rot,
                      args=(input[i], j, idx, parms["galois_keys"], parms["parms"], rot_dic['rot_lst'+ str(i)]))
        rot_workers.append(p)
        p.start()
for work in rot_workers:
    work.join()
    #inputs.append(rot_lst)
inputs = []

for i in range(input.__len__()):
    inputs.append(rot_dic['rot_lst'+ str(i)])

s3 = time.time()
print(f'Rotation: {s3-s1}')
"""


# In[ ]:


def multi_rot(x, idx, j, parms, gk, dic):
    context = SEALContext(parms)
    eva = Evaluator(context)
    y = eva.rotate_vector(x, idx[j - 1], gk)
    y.set_parms(parms)
    dic[j] = y


# In[ ]:


def convolution(input_f, conv_weight, conv_bias, s, parms):
    c_s = time.time()
    out_channels = []
    out_channel = conv_weight.__len__()
    in_channel = conv_weight[0].__len__()
    f = int(math.sqrt(conv_weight[0][0].__len__()))
    p = 0
    parms['n'] = int((parms['m'] + 2 * p - f) / s + 1)

    # search rotation index
    idx = count_rotation(parms['m'], f)
    for i in range(idx.__len__()):
        idx[i] = parms["valid_vector"][idx[i]]

    # valid values
    valid_index = []
    mask = [0 for i in range(parms['slots'])]
    for i in range(parms['n']):
        for j in range(parms['n']):
            valid_index.append(i * s * parms['m'] + j * s)
            mask[parms["valid_vector"][i * s * parms['m'] + j * s]] = 1
    mask = parms['encoder'].encode(mask, parms['scale'])
    mask.set_parms(parms["parms"])
    
    parms["galois_keys"].set_parms(parms["parms"])
    inputs = []
    rot_s = time.time()
    
    for i in range(in_channel):
        # dic and workers should not be per channel -> ?
        dic = mp.Manager().dict()
        workers = []
        for j in range(f*f):
            p = mp.Process(target=multi_rot,
                          args=(input_f[i], idx, j, parms["parms"], parms["galois_keys"], dic))
            workers.append(p)
            p.start()
        for work in workers:
            work.join()
        in_lst = []
        for k in range(f*f):
            dic[k].set_parms(parms["parms"])
            in_lst.append(dic[k])
        inputs.append(in_lst)
    """
    inputs = [[0 for i in range(f * f)] for j in range(in_channel)]
    for i in range(in_channel):
        for j in range(f * f):
            inputs[i][j] = parms['evaluator'].rotate_vector(input[i], idx[j - 1], parms['galois_keys'])
            inputs[i][j].set_parms(parms["parms"])
    """
    print(f"Rot time: {time.time() - rot_s}")
    # Process submit one by one 
    par_s = time.time()
    lst = mp.Manager().list()
    workers = []
    for i in range(out_channel):
        p = mp.Process(target=multi_conv,
                       args=(conv_weight[i], conv_bias[i], inputs, f, mask, parms["galois_keys"], parms["parms"],lst))
        workers.append(p)
        p.start()
    for work in workers:
        work.join() 
    print(f"Parallel time: {time.time() - par_s}")
    #lst_out = []
    #for x in lst:
    #    lst_out.append(pickle.loads(x))
    
    """
    # Pool submit one by one
    print("Parallel Start")
    pool = mp.Pool(processes = out_channel if out_channel < os.cpu_count() else os.cpu_count())    
    for i in range(out_channel):
        out_channels.append(
            pool.apply_async(multi_conv, 
                             (conv_weight[i], conv_bias[i], inputs, f, mask, parms["galois_keys"], parms["parms"])
                            ).get())  
    pool.close()
    pool.join()  
    
    """
    """
    # Concurrent submit one by one
    print("Parallel Start")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        res = []
        for i in range(out_channel):
            res.append(
            executor.submit(multi_conv, conv_weight[i], conv_bias[i], inputs, f, mask, parms["galois_keys"], parms["parms"]))
        for r in res:
            out_channels.append(r.result())
    print("Parallel End")
    """
    """
    # Pathos apipe
    print("Parallel Start")
    pool = pathos.multiprocessing.ProcessPool()
    for i in range(out_channel):
        out_channels.append(
            pool.apipe(
                multi_conv, conv_weight[i], conv_bias[i], inputs, f, mask, parms["galois_keys"], parms["parms"]))
    for i in range(out_channel):
        out_channels[i] = out_channels[i].get()
    print("Parallel End")
    """
    
    # set output of layer as input of next layer
    parms['m'] = parms['n']
    # set invalid vector
    Z = []
    for i in valid_index:
        Z.append(parms["valid_vector"][i])
    parms["valid_vector"] = Z
    c_e = time.time()
    conv_time = (c_e - c_s)/ out_channel
    print("Parallel End")
    return out_channels, parms, conv_time


# In[ ]:


def bn_act(x, b, c, parms):
    channels = len(x)
    X = []
    for i in range(channels):
        # HE square
        # x^2
        x_2 = parms['evaluator'].square(x[i])
        parms['evaluator'].relinearize_inplace(x_2, parms['relin_keys'])
        parms['evaluator'].rescale_to_next_inplace(x_2)

        # b * x
        b_prime = parms['encoder'].encode(b[i].item(), parms['scale'])
        parms['evaluator'].mod_switch_to_inplace(b_prime, x[i].parms_id())
        b_x = parms['evaluator'].multiply_plain(x[i], b_prime)
        parms['evaluator'].rescale_to_next_inplace(b_x)

        c_prime = parms['encoder'].encode(c[i].item(), parms['scale'])

        parms['evaluator'].mod_switch_to_inplace(c_prime, x_2.parms_id())
        parms['evaluator'].mod_switch_to_inplace(b_x, x_2.parms_id())

        x_2.scale(2 ** int(math.log2(c_prime.scale())))
        b_x.scale(2 ** int(math.log2(c_prime.scale())))

        X.append(parms['evaluator'].add_plain(parms['evaluator'].add(x_2, b_x), c_prime))
    return X


# In[ ]:


def multi_pack(x, i, vv, gk, n, rot, slot, parms, scale, lst):
    # HE rotation
    # idx: valuable idx; rot_idx: rotation index
    # generate all-zeroes mask
    context = SEALContext(parms)
    eva = Evaluator(context)
    encoder = CKKSEncoder(context)
    
    mask = [0 for h in range(slot)]
    mask[0] = 1
    mask = encoder.encode(mask, scale)
    eva.mod_switch_to_inplace(mask, x.parms_id())
    Z = []
    Z.append(eva.multiply_plain(x, mask))
    for k in range(n * n - 1):
        # HE element-wise multiplication and addition
        mask = [0 for h in range(slot)]
        mask[vv[k + 1]] = 1
        mask = encoder.encode(mask, scale)
        eva.mod_switch_to_inplace(mask, x.parms_id())
        Z.append(
            eva.rotate_vector(
                eva.multiply_plain(x, mask),
            rot[k], gk
            )
        )
    Z = eva.add_many(Z)
    Z = eva.rotate_vector(Z, -(n * n * i), gk)
    Z.set_parms(parms)
    lst.append(pickle.dumps(Z))
    # return parms['evaluator'].rotate_vector(x, -(n * n * i), gk)

def pack_vectors(x, parms):
    pv_s = time.time()
    channels = x.__len__()
    Z = []
    n = parms['n']
    # Y.append(x[0])
    rot_idx = []
    for j in range(1, n * n):
        rot_idx.append(parms["valid_vector"][j] - j)
    
    lst = mp.Manager().list()
    workers = []
    parms["galois_keys"].set_parms(parms["parms"])
    
    for i in range(channels):
        x[i].set_parms(parms['parms'])
        p = mp.Process(target=multi_pack,
                       args=(x[i], i, parms["valid_vector"], parms["galois_keys"],
                             parms['n'], rot_idx, parms['slots'], parms["parms"], parms['scale'], lst))
        workers.append(p)
        p.start()
    for work in workers:
        work.join()
    lst_out = []
    for x in lst:
        lst_out.append(pickle.loads(x))
    
    Z = parms['evaluator'].add_many(lst_out)
    parms['evaluator'].rescale_to_next_inplace(Z)
    pv_e = time.time()
    pack_time = (pv_e - pv_s)/channels
    return Z,pack_time


# In[ ]:


def multi_fc(x, w, i, channel, threshold, gk, parms, lst):
    context = SEALContext(parms)
    eva = Evaluator(context)
    if i == 0:
        temp = x
    if i > 0 and i < threshold:
        temp = eva.rotate_vector(x, i, gk)        
    if i>= threshold:
        B = []
        B.append(eva.rotate_vector(x, i, gk))
        B.append(eva.rotate_vector(x, -channel + i, gk))
        temp = eva.add_many(B)     
    y = eva.multiply_plain(temp, w)
    y.set_parms(parms)
    lst.append(pickle.dumps(y))    


# In[ ]:


def fully_connected(x, weight, bias, parms):
    fc_s = time.time()
    Y = []
    temp = x
    channels  = weight.__len__()
    rot_single = channels - bias.__len__() + 1
    
    lst = mp.Manager().list()
    workers = []
    parms["galois_keys"].set_parms(parms["parms"])
    x.set_parms(parms['parms'])
    
    for i in range(channels):
        w = parms['encoder'].encode(weight[i], parms['scale'])
        parms['evaluator'].mod_switch_to_inplace(w, x.parms_id())
        w.set_parms(parms['parms'])
        p = mp.Process(target=multi_fc,
                       args=(x, w, i, channels, rot_single, parms["galois_keys"], parms["parms"], lst))
        workers.append(p)
        p.start()
    for work in workers:
        work.join()
    lst_out = []
    for x in lst:
        lst_out.append(pickle.loads(x))  
                  
    Y = parms['evaluator'].add_many(lst_out)
    parms['evaluator'].rescale_to_next_inplace(Y)
    B = parms['encoder'].encode(bias, parms['scale'])
    parms['evaluator'].mod_switch_to_inplace(B, Y.parms_id())
    Y.scale(2 ** int(math.log2(B.scale())))
    fc_e = time.time()
    fc_time = (fc_e - fc_s)/channels
    return parms['evaluator'].add_plain(Y, B), fc_time

