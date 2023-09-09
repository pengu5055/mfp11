import numpy as np
from scipy import special

def scalar_product3(m, n):
    return -2/(2*m + 1) * special.beta(2*m + 3, n + 1)


def vector(M, N):
    ms = range(M)
    ns = range(1, N+1)
    b = np.zeros(M*N)
    for i in range(M):
        m = ms[i]
        for j in range(N):
            n = ns[j]
            b[i*N + j] = scalar_product3(m, n)
    return b

print(vector(2, 5))