import numpy as np
import scipy
from ort_poly2 import nquad
import math

## Kolmogorov-Smirnov test Critical Value

def Kolmogorov_Smirnov_CV(alpha, sample_size):
    c_alpha = np.sqrt(-np.log(alpha/2) * 0.5)
    factor = np.sqrt(2 / sample_size)
    return np.sqrt(-(1 / sample_size) * np.log(alpha / 2)), c_alpha, factor

def empirical_cdf(list_val, n_dots):
    dots = np.linspace(min(list_val), max(list_val), n_dots)
    res = []
    for j in dots:
        res.append(len([i for i in list_val if i < j]) / len(list_val))
    return res


def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def Uniform(a, b, size = 1):
    return (a + np.random.rand(size) * (b-a))


def N(mu, sigma_sq, size = 1):  ## sigma_sq = Variance!
    return np.random.normal(mu, np.sqrt(sigma_sq), size)

def Beta(a, b, size = 1):
    return np.random.beta(a, b, size)

def G(a, b, size = 1):
    return np.random.gamma(a, 1/b, size)

def Exp(a, size = 1):
    return np.random.exponential(1/a, size)

def Bernoulli(p, size = 1):
    return np.random.binomial(1, p, size)


def trunc_Exp(a, _a, _b, size = 1):
    res = np.zeros(shape=(0))
    diff_size = size

    while diff_size > 0:
        tmp = np.random.exponential(1 / a, size=diff_size)
        cur_size = diff_size
        tmp = tmp[(tmp >= _a) & (tmp <= _b)]
        diff_size = cur_size - tmp.shape[0]
        res = np.hstack([res, tmp])

    np.random.shuffle(res)
    return res


def trunc_norm(mu, sigma, a, b, size = 1):
    res = np.zeros(shape=(0))
    diff_size = size

    while diff_size > 0:
        tmp = N(mu, sigma**2, size = diff_size)
        cur_size = diff_size
        tmp = tmp[(tmp >= a) & (tmp <= b)]
        diff_size = cur_size - tmp.shape[0]
        res = np.hstack([res, tmp])

    np.random.shuffle(res)
    return res




def trunc_Expon_pdf(x, _l, _a, _b):
    if x < _a or x > _b:
        return 0
    tmp = lambda x: scipy.stats.gamma.pdf(x, a=1, scale=1/_l)

    m1 = nquad(lambda x: tmp(x),
                [[0,_a]],
                full_output=False)[0]
    m2 = nquad(lambda x: tmp(x),
               [[0, _b]],
               full_output=False)[0]
    return tmp(x) / (m2 - m1)



def Irwin_Hall_pdf(x, n):
    if x < 0 or x > n:
        return 0

    pdf = 0
    for k in range(int(math.floor(x)) + 1):
        term = (-1) ** k * math.comb(n, k) * (x - k) ** (n - 1)
        pdf += term

    return pdf / (math.factorial(n - 1))
