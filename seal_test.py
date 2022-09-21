import sys
import time
import math
import numpy as np
import seal
from seal import scheme_type


def print_example_banner(title):
    title_length = len(title)
    banner_length = title_length + 2 * 10
    banner_top = '+' + '-' * (banner_length - 2) + '+'
    banner_middle = '|' + ' ' * 9 + title + ' ' * 9 + '|'
    print(banner_top)
    print(banner_middle)
    print(banner_top)


def print_parameters(context):
    context_data = context.key_context_data()
    if context_data.parms().scheme() == scheme_type.bfv:
        scheme_name = 'bfv'
    elif context_data.parms().scheme() == scheme_type.ckks:
        scheme_name = 'ckks'
    else:
        scheme_name = 'none'
    print('/')
    print('| Encryption parameters')
    print('| scheme: ' + scheme_name)
    print(f'| poly_modulus_degree: {context_data.parms().poly_modulus_degree()}')
    coeff_modulus = context_data.parms().coeff_modulus()
    coeff_modulus_sum = 0
    for j in coeff_modulus:
        coeff_modulus_sum += j.bit_count()
    print(f'| coeff_modulus size: {coeff_modulus_sum}(', end='')
    for i in range(len(coeff_modulus) - 1):
        print(f'{coeff_modulus[i].bit_count()} + ', end='')
    print(f'{coeff_modulus[-1].bit_count()}) bits')
    if context_data.parms().scheme() == scheme_type.bfv:
        print(f'| plain_modulus: {context_data.parms().plain_modulus().value()}')
    print('\\')


def print_vector(vec, print_size=4, prec=3):
    slot_count = len(vec)
    print()
    if slot_count <= 2*print_size:
        print('    [', end='')
        for i in range(slot_count):
            print(f' {vec[i]:.{prec}f}' + (',' if (i != slot_count - 1) else ' ]\n'), end='')
    else:
        print('    [', end='')
        for i in range(print_size):
            print(f' {vec[i]:.{prec}f},', end='')
        if slot_count > 2*print_size:
            print(' ...,', end='')
        for i in range(slot_count - print_size, slot_count):
            print(f' {vec[i]:.{prec}f}' + (',' if (i != slot_count - 1) else ' ]\n'), end='')
    print()


def main():
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 16384
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(
        poly_modulus_degree, [60, 40, 40, 40, 40, 60]))
    scale = 2.0 ** 40
    context = SEALContext(parms)
    print_parameters(context)

    ckks_encoder = CKKSEncoder(context)
    slot_count = ckks_encoder.slot_count()
    print(f'Number of slots: {slot_count}')

    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()
    galois_keys = keygen.create_galois_keys()

    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)

    plaintext_a = [1, 2, 3, 4, 5]
    plaintext_b = [2, 4, 6, 8, 10]

    print("Plaintext A:{} \n Plaintext B:{}".format(plaintext_a, plaintext_b))
    print(np.sum(plaintext_a, plaintext_b))
    print(np.dot(plaintext_a, plaintext_b))

    plain_a = ckks_encoder.encode(plaintext_a.flatten(), scale)
    plain_b = ckks_encoder.encode(plaintext_b.flatten(), scale)
    ciphertext_a = encryptor.encrypt(plain_a)
    ciphertext_b = encryptor.encrypt(plain_b)
    # plaintext + ciphertext
    ciphertext_c = ciphertext_a + plain_a
    plain_c = decryptor.decrypt(ciphertext_c)
    plaintext_c = ckks_encoder.decode(plain_c)
    print(plaintext_c)

    # plaintext * ciphertext
    ciphertext_c = np.dot(ciphertext_a, plain_a)
    plain_c = decryptor.decrypt(ciphertext_c)
    plaintext_c = ckks_encoder.decode(plain_c)
    print(plaintext_c)

    # ciphertext + ciphertext
    ciphertext_c = ciphertext_a + ciphertext_b
    plain_c = decryptor.decrypt(ciphertext_c)
    plaintext_c = ckks_encoder.decode(plain_c)
    print(plaintext_c)

    # ciphertext * ciphertext
    ciphertext_c = np.dot(ciphertext_a, ciphertext_b)
    plain_c = decryptor.decrypt(ciphertext_c)
    plaintext_c = ckks_encoder.decode(plain_c)
    print(plaintext_c)

if __name__ == '__main__':
    main()