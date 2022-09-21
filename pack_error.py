import numpy as np
from seal import *
import matplotlib.pyplot as plt

def Encryption_Parameters():
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    bit_scale = 30
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, [60, bit_scale, 60]))
    context = SEALContext(parms)
    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()

    parms = {
        'context' : context,
        'encryptor': Encryptor(context, public_key),
        'evaluator': Evaluator(context),
        'decryptor': Decryptor(context, secret_key),
        'encoder': CKKSEncoder(context),
        'scale': 2.0 ** bit_scale,
        'galois_keys': keygen.create_galois_keys(),
        'relin_keys' : keygen.create_relin_keys(),
        'slots': CKKSEncoder(context).slot_count()
    }
    return parms

def pack_vactors(x, n):
    channels = x.__len__()
    Y = []
    Y.append(x[0])
    for i in range(1, channels):
        # HE rotation
        Y.append(parms['evaluator'].rotate_vector(x[i], -(n * i), parms['galois_keys']))
    Y = parms['evaluator'].add_many(Y)
    return Y


if __name__=='__main__':

    parms = Encryption_Parameters()
    order = np.arange(2, 51, 1)
    error = []
    for i in range (len(order)):
        n = order[i]
        X = np.random.RandomState().uniform(-1, 1, (n, n))
        X_squeeze = X.reshape(-1)

        X = X.tolist()
        Y = []
        for j in range(n):
            Y.append(parms['encryptor'].encrypt(
                parms['encoder'].encode(X[j], parms['scale'])))
        Y_squeeze = pack_vactors(Y, n)
        Y_squeeze = parms['encoder'].decode(parms['decryptor'].decrypt(Y_squeeze))[:n*n]
        Y_squeeze = np.array(Y_squeeze)
        error_array = np.abs(X_squeeze - Y_squeeze)

        error_array = error_array / np.abs(X_squeeze)
        error_sum = np.sum(error_array)
        error.append(error_sum/error_array.size)

    plt.figure()
    plt.plot(order, error, 'b')
    plt.plot(order, error, 'ro')
    plt.grid(True)
    plt.axis('tight')
    plt.xlabel('Order')
    plt.xticks(order)
    plt.ylabel('Error')
    plt.title('Error of Pack Vectors')
    plt.show()

    # print(error)