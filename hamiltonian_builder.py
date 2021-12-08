import numpy as np
import math

def dot(S, x):
    sum = 0
    for i in S:
        sum += (x >> i) & 1
    return sum

def my_fe(x):
    N = len(x)
    n = int(np.log2(N))
    def ret_func(S, frac=False):
        if frac:
            nom = np.sum(list(map(
                lambda i: x[i] * (-1)**dot(n-np.array(S)-1, i),
                range(0,N))))
            dnom = N
            d = np.gcd(nom,dnom)
            return (nom//d, dnom//d)
        else:
            return (1/N * 
                np.sum(list(map(
                    lambda i: x[i] * (-1)**dot(n-np.array(S)-1, i),
                    range(0,N)))
                )
            )
    return ret_func

def sort_meth(e, n):
    return len(e)*(n**n)+np.sum([(n**(len(e)-i-1)) * v for (i,v) in enumerate(e)])

def build_hamiltonian(x):
    s = ""
    f_hat = my_fe(x)
    N = len(x)
    n = int(np.log2(N))
    subsets = [[j for j in range(n) if (i & (1 << j))] for i in range(1 << n)]
    subsets.sort(key=lambda e: sort_meth(e, n))
    for subset in subsets:
        # w_k = f_hat(subset)
        # if w_k != 0:
        #     s_z = " ".join([f"Z_{{{i}}}" for i in subset])  if len(subset) != 0 else "I"
        #     ir = (w_k).as_integer_ratio()
        #     s += f"\\frac{{{ir[0]}}}{{{ir[1]}}} {s_z} + "
        ir = f_hat(subset, frac=True)
        if ir[0] != 0:
            s_z = " ".join([f"Z_{{{i}}}" for i in subset])  if len(subset) != 0 else "I"
            s += f"\\frac{{{ir[0]}}}{{{ir[1]}}} {s_z} + "

    return s

def gen_happy_array(d):
    happy_arr = []
    for i in range(2**(d+1)):
        num_diff = 0
        num_same = 0
        v = dot([d], i)
        for j in range(d):
            u = dot([j], i)
            if u == v: num_same += 1
            else: num_diff += 1
        happy_arr.append(1 if num_diff >= num_same else 0)
    return happy_arr


# d=2
# 000, 001, 010, 011, 100, 101, 110, 111
#  0    1    1    1    1    1    1    0

# d=3
# 0000, 0001, 0010, 0011, 0100, 0101, 0110, 0111
#  0     0     0     1     0     1     1     1
# 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111
#  1     1     1     0     1     0     0     0

# d=4
# 00000, 00001, 00010, 00011, 00100, 00101, 00110, 00111
#   0      0      0      1      0      1      1      1
# 01000, 01001, 01010, 01011, 01100, 01101, 01110, 01111
#   0      1      1      1      1      1      1      1
# 10000, 10001, 10010, 10011, 10100, 10101, 10110, 10111
#   1      1      1      1      1      1      1      0
# 11000, 11001, 11010, 11011, 11100, 11101, 11110, 11111
#   1      1      1      0      1      0      0      0

if __name__ == "__main__":
    # for i in range(2, 21, 2):
    #     arr = gen_happy_array(i)
    #     (n,d) = my_fe(arr)([], frac=True)
    #     print(f"{i}, {n/d}, {n}/{d}")

    print(build_hamiltonian(gen_happy_array(2)), "\n")

    print(build_hamiltonian(gen_happy_array(3)), "\n")

    print(build_hamiltonian(gen_happy_array(4)), "\n")

    # print(build_hamiltonian(gen_happy_array(5)), "\n")

    # print(build_hamiltonian(gen_happy_array(6)), "\n")

    # print(build_hamiltonian(gen_happy_array(7)), "\n")

    # print(build_hamiltonian(gen_happy_array(8)), "\n")

