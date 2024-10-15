import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as font_manager
from statistics import median, mode, mean, variance
from scipy.stats import skew, kurtosis
import os
from mpmath import nsum
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick
import matplotlib.font_manager as font_manager
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
import numpy as np
import sympy
from scipy.stats import expon
import statsmodels.api as sm
import scipy
import math
from ort_poly2 import *
from Bivariate_normal_distribution import trunc_Bivariate_normal_dist



#######################################################################################

def Kolmogorov_Smirnov_CV(alpha, sample_size):
    c_alpha = np.sqrt(-np.log(alpha/2) * 0.5)
    factor = np.sqrt(2 / sample_size)
    return np.sqrt(-(1 / sample_size) * np.log(alpha / 2)), c_alpha, factor

def empirical_cdf(list_val, n_dots):
    dots = np.linspace(min(list_val), max(list_val), n_dots)
    res = []
    for j in dots:
        #res.append(sum([i < j for i in list_val]) / len(list_val))
        res.append(len([i for i in list_val if i < j]) / len(list_val))
    return res

def laplace(x, mu, b):
    return (b / 2)*np.exp(-b * np.abs(x - mu))


def funny(x, mu, sigma, a1 = 0.34, a2 = 0.33, a3 = 0.33):

    t1 = norm_pdf(x, mu, sigma, infty=True)
    t2 = laplace(x, mu, math.sqrt(2)/sigma)
    t3 = scipy.stats.gamma.pdf(x, a=mu**2 / sigma**2, scale=sigma**2 / mu)

    return a1 * t1 + a2 * t2 * a3 * t3


def multivar_poly_frame(coef_mat: dict,
                        comb: list,
                        moms: list
                        ) -> (float, pd.DataFrame):
    res = pd.DataFrame({0 : [1]})
    for j in list(coef_mat.keys())[::-1]: ### NEED TO BE FIXED!
        tmp = pd.DataFrame({0 : [1]*coef_mat[j].shape[1],
                            j : list(coef_mat[j][comb[j-1]])}
                           )
        res = res.merge(tmp, on = 0, how = 'outer')
    res["mom"] = moms
    res.drop(0, axis = 1, inplace = True)
    res['prod'] = 1.0
    for j in res.columns:
        if j != 'prod':
            res['prod'] = [a * b for a, b in zip(res['prod'],res[j])]
    coef = sum(res['prod']) ## RESULTING COEFFICIENT ##
    res.drop('prod', axis=1, inplace=True)
    return coef, res

def est1(*args, est_poly, mes):
    res1 = est_poly
    res2 = 1.0
    for num, j in enumerate(args):
        res2 *= mes[num+1](j)
        res1 = res1.subs(vars_[num + 1], j)
    return res1 * res2
#######################################################################################

def logistic(x):
    return 1 / (1 + np.exp(-x))

def trunc_norm(mu, sigma, a, b):
    tmp = np.random.normal(mu, sigma)
    if tmp < a or tmp > b:
        return trunc_norm(mu, sigma, a, b)
    return tmp

def trunc_Expon(x, _l, _a, _b):
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

def Expon(x, _l):
    if x < 0:
        return 0
    return _l * math.exp(-_l * x)


def trunc_Expon_rev(x, _l, _a, _b):
    if x < _a or x > _b:
        return 0
    tmp = lambda x: scipy.stats.gamma.pdf(-(x-_b), a=1, scale=1/_l)

    m1 = nquad(lambda x: tmp(x),
                [[_a,_b]],
                full_output=False)[0]
    return tmp(x) / m1




def trunc_Gamma_rev(x, k, _l, _a, _b):
    if x < _a or x > _b:
        return 0
    tmp = lambda x: scipy.stats.gamma.pdf(-(x-_b), a=k, scale=1/_l)

    m1 = nquad(lambda x: tmp(x),
                [[_a, _b]],
                full_output=False)[0]

    return tmp(x) / m1


def Gamma_rev(x, k, _l, _b = 0):
    if x > _b:
        return 0
    return scipy.stats.gamma.pdf(-(x-_b), a=k, scale=1/_l)


def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def Uniform(a, b):
    return (a + np.random.rand(1) * (b-a))[0]

def Categorical(*args):   # from 0
    q = np.linspace(0, 1, len(args)+1)
    for i, j in enumerate(args):
        q[i+1] = q[i] + j

    val = np.random.rand(1)[0]
    for i, j in enumerate(q):
        if val < j:
            return i-1

def N(mu, sigma_sq):
    return np.random.normal(mu, np.sqrt(sigma_sq))

def Beta(a, b):
    return np.random.beta(a, b)

def G(a, b):
    return np.random.gamma(a, 1/b)

def Exp(a):
    return np.random.exponential(1/a)

def Bernoulli(p):
    return np.random.binomial(1, p, 1)[0]


def trunc_Exp(a, _a, _b):
    tmp = np.random.exponential(1/a)

    if tmp < _a or tmp > _b:
        return trunc_Exp(a, _a, _b)
    return tmp


#########################################################################################
#########################################################################################

my_path = os.path.abspath('Figures')

#########################################################################################
#########################################################################################
#########################################################################################
###################  Vasicek model   #############################
#########################################################################################
#########################################################################################
#########################################################################################

n_moms = 4
num_it = 100
num_rep = 1000000
r_list = []

r_moms = [0] * (1 + n_moms)**2

for i in tqdm(range(num_rep)):
    a = 0.5
    b = 0.02
    sigma = 0.2
    w = 0
    r = 0.08
    for j in range(1, num_it + 1):
        w = trunc_norm(0, 1, -10, 10)
        r = (1 - a) * r + a * b + sigma * w
    r_list.append(r)

plt.hist(r_list, bins = 40, density = True)

r_moms = [1]

for j in range(1, 27):
    r_moms.append(sum([k ** j for k in r_list]) / num_rep)


n_moms = len(r_moms[:(n_moms+1)]) - 1

n_moms = 2
moms = [float(j) for j in r_moms[:(n_moms+1)]]
work_list = [float(j) for j in r_list]

params = {}
params[1] = {}
params[1]['start'] = -math.inf#min(work_list)#-300
params[1]['stop'] = math.inf#max(work_list)#300
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'], infty = True)


mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = True)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)

size = 1000
x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=18)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "r histogram"], prop = font1,  frameon=False)
plt.savefig(my_path + "/r_estimator_Vasicek.jpeg")




cdf_est = [0] * len(x_tmp)
r_list = sorted([float(q) for q in r_list])
cdf_true = empirical_cdf([float(q) for q in r_list], len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)




#########################################################################################
#########################################################################################
#########################################################################################
##############  Stuttering P   ########################
#########################################################################################
#########################################################################################
#########################################################################################

n_moms = 5
num_it = 10
num_rep = 1000000
s_list = []


for i in tqdm(range(num_rep)):
    f = 0
    x = -1
    y = 1
    s = 0
    p = 0.75
    for j in range(1, num_it + 1): #while true:
        u1 = Uniform(0, 2)
        u2 = Uniform(0, 4)
        f = Bernoulli(p)
        x = x + f * u1
        y = y + f * u2
        s = x + y
    s_list.append(s)

work_list_tmp = sorted(s_list)

plt.hist(s_list, bins = 50, density = True)

s_moms = [1]

for j in range(1, 27):
    s_moms.append(sum([k ** j for k in s_list]) / num_rep)


n_moms = len(s_moms[:(n_moms+1)]) - 1

n_moms = 2
moms = [float(j) for j in s_moms[:(n_moms+1)]]
work_list = [float(j) for j in s_list]

params = {}
params[1] = {}
params[1]['start'] = -math.inf
params[1]['stop'] = math.inf
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'], infty = True)

mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = True)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)

size = 1000
x_tmp_l = np.linspace(min(s_list), max(s_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]
font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=18)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "S histogram"], prop = font1,  frameon=False)
plt.savefig(my_path + "/Stuttering_P.jpeg")




cdf_est = [0] * len(x_tmp)
work_list = sorted([float(q) for q in work_list])
cdf_true = empirical_cdf([float(q) for q in work_list], len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)

plt.plot(x_tmp, cdf_true, x_tmp, cdf_est)





#########################################################################################
#########################################################################################
#########################################################################################
##############  1D Random Walk   ########################
#########################################################################################
#########################################################################################
#########################################################################################

n_moms = 5
num_it = 100
num_rep = 1000000
x_list = []


for i in tqdm(range(num_rep)):
    p = 0.5
    x = 2
    for j in range(1, num_it + 1):
        f = Bernoulli(p)
        c1 = x - 1
        c2 = x + 1
        x = c1 * f + c2 * (1 - f)
    x_list.append(x)

work_list_tmp = sorted(x_list)
work_unique = list(set(x_list))

plt.hist(x_list, bins = len(work_unique), density = True)

x_moms = [1]

for j in range(1, 27):
    x_moms.append(sum([k ** j for k in x_list]) / num_rep)



n_moms = len(x_moms[:(n_moms+1)]) - 1

n_moms = 2
moms = [float(j) for j in x_moms[:(n_moms+1)]]
work_list = [float(j) for j in x_list]

params = {}
params[1] = {}
params[1]['start'] = -math.inf#min(work_list)#-98
params[1]['stop'] = math.inf#max(work_list)#102
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'], infty = True)


mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = True)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)

size = 1000

x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=18)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = len(set(work_list)), density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "X histogram"], prop = font1,  frameon=False, loc = 'upper left')
plt.savefig(my_path + "/1D_random_walk.jpeg")




cdf_est = [0] * len(x_tmp)
work_list = sorted([float(q) for q in work_list])
cdf_true = empirical_cdf([float(q) for q in work_list], len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)


plt.plot(x_tmp, cdf_true, x_tmp, cdf_est)





#########################################################################################
#########################################################################################
#########################################################################################
##############  2D Random Walk   ########################
#########################################################################################
#########################################################################################
#########################################################################################

n_moms = 2
num_it = 100
num_rep = 1000000
x_list = []
y_list = []

xy_moms = [0] * (1 + n_moms)**2

for i in tqdm(range(num_rep)):
    h = 0
    x = 0
    y = 0
    for j in range(1, num_it + 1): #while true:
        h = Bernoulli(0.5)
        h1 = Bernoulli(0.5)
        h2 = Bernoulli(0.5)
        x = (x - h) * h1 +  (x + h) * (1 - h1)
        y = (y + (1 - h)) * h2 + (y - (1 - h)) * (1 - h2)
    x_list.append(x)
    y_list.append(y)
    for m in range(0, n_moms + 1):
        for n in range(0, n_moms + 1):
            xy_moms[m * (n_moms + 1) + n] += (y ** m) * (x ** n)
xy_moms = [i / num_rep for i in xy_moms]


work_list_tmp_x = sorted(x_list)
work_unique_x = list(set(x_list))

work_list_tmp_y = sorted(y_list)
work_unique_y = list(set(y_list))

plt.hist(x_list, bins = len(work_unique_x), density = True)
plt.hist(y_list, bins = len(work_unique_y), density = True)

x_moms = [1]
y_moms = [1]

for j in range(1, 27):
    x_moms.append(sum([k ** j for k in x_list]) / num_rep)
    y_moms.append(sum([k ** j for k in y_list]) / num_rep)


n_moms = len(x_moms[:(n_moms+1)]) - 1


moms = [float(j) for j in y_moms[:(n_moms+1)]]
work_list = [float(j) for j in y_list]

params = {}
params[1] = {}
params[1]['start'] = -100#-100#min(work_list)#-98
params[1]['stop'] = 100#100#max(work_list)#102
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):

    return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'], infty = True)


mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = True)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)


#def est(*args):
#    return scipy.stats.norm.pdf(*args, moms[1], np.sqrt(moms[2] - moms[1]**2))


size = 1000
#x_tmp_r = np.linspace(params[1]['start'], params[1]['stop'], size)
x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=18)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = len(set(work_list)), density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "Y histogram"], prop = font1,  frameon=False, loc = 'upper left')
plt.savefig(my_path + "/Y_2D_random_walk.jpeg")




cdf_est = [0] * len(x_tmp)
work_list = sorted([float(q) for q in work_list])
cdf_true = empirical_cdf([float(q) for q in work_list], len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)


plt.plot(x_tmp, cdf_true, x_tmp, cdf_est)




moms = [float(j) for j in x_moms[:(n_moms+1)]]
work_list = [float(j) for j in x_list]

params = {}
params[1] = {}
params[1]['start'] = -100#
params[1]['stop'] = 100
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):

    return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'], infty = True)


mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = True)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)



size = 1000
#x_tmp_r = np.linspace(params[1]['start'], params[1]['stop'], size)
x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=18)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = len(set(work_list)), density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "X histogram"], prop = font1,  frameon=False, loc = 'upper left')
plt.savefig(my_path + "/X_2D_random_walk.jpeg")




cdf_est = [0] * len(x_tmp)
work_list = sorted([float(q) for q in work_list])
cdf_true = empirical_cdf([float(q) for q in work_list], len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)


plt.plot(x_tmp, cdf_true, x_tmp, cdf_est)


moms = xy_moms

params = {}
params[1] = {}
params[1]['start'] = -100#min(x_list)
params[1]['stop'] = 100#max(x_list)
params[1]['opts'] = {'epsabs' : 1.49e-7}

params[2] = {}
params[2]['start'] = -100#min(y_list)
params[2]['stop'] = 100#max(y_list)
params[2]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, x_moms[1], np.sqrt(x_moms[2] - x_moms[1] ** 2),
                          params[1]['start'], params[1]['stop'])

def mes2(x1):
    return trunc_norm_pdf(x1, y_moms[1], np.sqrt(y_moms[2] - y_moms[1] ** 2),
                          params[2]['start'], params[2]['stop'])

mes = dict()
mes[1] = mes1
mes[2] = mes2

X = K(params, mes, max_poly_degree=n_moms, infty = False)
X.create_orthogonal_poly_collection()
X.create_poly_combs()


vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0


tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms)


for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                              list(X.poly_combs.loc[line, :]),
                              moms)[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()


def est(*args):
    res1 = est_poly
    res2 = 1.0
    for num, j in enumerate(args):
        res2 *= mes[num+1](j)
        res1 = res1.subs(vars_[num + 1], j)
    return res1 * res2


fig = plt.figure(figsize=(10, 8))

size = 100
ax = fig.add_subplot(projection='3d')

X_tmp = np.linspace(min(x_list), max(x_list), size)
Y_tmp = np.linspace(min(y_list), max(y_list), size)

X_tmp, Y_tmp = np.meshgrid(X_tmp, Y_tmp)
R = np.array([est(i, j) for i, j in zip(np.ravel(X_tmp), np.ravel(Y_tmp))])
Z = R.reshape(X_tmp.shape)

surf = ax.plot_surface(X_tmp, Y_tmp, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
cbar.ax.tick_params(labelsize=15)

plt.xticks(fontsize=10, weight = 'bold')
plt.yticks(fontsize=10, weight = 'bold')
ax.set_xlabel('$X$', fontsize=20, rotation=160, weight = 'bold')
ax.set_ylabel('$Y$', fontsize=20, weight = 'bold')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('$\hat{f}$', fontsize=20, rotation = 0, weight = 'bold')
for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)
for t in ax.zaxis.get_major_ticks(): t.label.set_weight('bold')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#plt.legend(["Bivariate K-series"], prop = font1,  frameon=False)
print("Min: ", Z.min(), ", Max: ", Z.max())
plt.savefig(my_path + "/estimator_2D_random_walk.jpeg")


#######################           Sampling         #####################################

_size = 100
X_tmp = np.linspace(min(x_list), max(x_list), _size)
Y_tmp = np.linspace(min(y_list), max(y_list), _size)
Z_vals = np.zeros((_size, _size))

for i, y_val in enumerate(Y_tmp):
    for j, x_val in enumerate(X_tmp):
        Z_vals[i, j] = 500000 * est(x_val, y_val)


Z_x_list = []
Z_y_list = []

for i in tqdm(range(Y_tmp.shape[0])):
    for j in range(X_tmp.shape[0]):
        for k in range(int(Z_vals[i, j])):
            Z_x_list.append(X_tmp[j])
            Z_y_list.append(Y_tmp[i])


plt.hist(Z_x_list, bins = 50, density = True); plt.hist(x_list, bins = 50, density = True); plt.legend(['ést', 'true',])
plt.hist(Z_y_list, bins = 50, density = True); plt.hist(y_list, bins = 50, density = True)

a = np.column_stack((Z_x_list, Z_y_list))
b = np.column_stack((x_list, y_list))

pd.DataFrame({'X_est' : a[:, 0],'Y_est' : a[:, 1]}).to_csv('2d_RW_estim.csv')
pd.DataFrame({ 'X_true' : b[:, 0],'Y_true' : b[:, 1]}).to_csv('2d_RW_true.csv')


tmp_step_x = 15
tmp_step_y = 15

t1 = np.array(np.arange(params[1]['start'], params[1]['stop'] + tmp_step_x, tmp_step_x))
t2 = np.array(np.arange(params[2]['start'], params[2]['stop'] + tmp_step_y, tmp_step_y))

step1 = tmp_step_x
step2 = tmp_step_y

num_el = {}
num_el1 = {}
num_el_list = []
num_el_list1 = []
ind = []
for j in t1:

    for i in t2:

        ind.append('[' + str(round(j, 1)) + "," + str(round(i, 1)) + '] - [' + str(round(j + step1, 1)) + "," + str(round(i, 1)) + ']\n' +
                   '[' + str(round(j, 1)) + ", " + str(round(i + step2, 1)) + '] - [' + str(round(j + step1, 1)) + ", " + str(round(i + step2, 1)) + ']'
                   )
        num_el_list.append(a[np.logical_and(np.logical_and(a[:, 0] <= j + step1, a[:, 1] <= i + step2),
                                        np.logical_and(a[:, 0] > j, a[:, 1] > i)
                                        )].shape[0])
        num_el_list1.append(b[np.logical_and(np.logical_and(b[:, 0] <= j + step1, b[:, 1] <= i + step2),
                                         np.logical_and(b[:, 0] > j, b[:, 1] > i)
                                         )].shape[0])
trsh = 500
ind = [ind[i] for i in np.where(np.array(num_el_list1) > trsh)[0]]
num_el_list = [num_el_list[i] for i in np.where(np.array(num_el_list1) > trsh)[0]]
print(sum(num_el_list))
num_el_list = [i * 100 / sum(num_el_list) for i in num_el_list]
num_el_list1 = [num_el_list1[i]  for i in np.where(np.array(num_el_list1) > trsh)[0]]
print(sum(num_el_list1))
num_el_list1 = [i * 100 / sum(num_el_list1) for i in num_el_list1]

data = pd.DataFrame({'Estimated': num_el_list, 'True' : num_el_list1}, index = ind)
data['Rectangle'] = data.index


font = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=16)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
fp = data.plot(x = 'Rectangle', y = ['Estimated', 'True'], kind = 'bar', color = ['red', 'blue'],
               width = 0.9, position = 0.5, figsize=(10,7))
plt.tight_layout(pad = 4)
fp.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
plt.legend(prop = font, loc = 'upper left',  frameon=False)
plt.ylabel('Percentage of points per region', labelpad = 5, fontsize = 15, weight = 'bold')
plt.xlabel('', labelpad = 15, fontsize = 10, weight = 'bold')
plt.xticks(rotation=55, ha='right', weight = 'bold')
plt.yticks(fontsize = 12, weight = 'bold')
plt.savefig(my_path + "/comparison_2D_random_walk.jpeg")



#########################################################################################
##############  PDP  ########################
#########################################################################################
#########################################################################################
#########################################################################################

n_moms = 6
num_it = 100
num_rep = 1000000
x_list = []
y_list = []
xy_moms = [0] * (1 + n_moms)**2


for i in tqdm(range(num_rep)):
    k1 = 4
    k2 = 40
    a = 0.2
    b = 4
    p = 0.5
    rho = 0.5
    y = 0
    x = 0
    for j in range(1, num_it + 1): #while true:
        c1 = Bernoulli(p)
        k = c1 * k1 + (1-c1) * k2
        y, x = (1 - rho) * y + k, (1 - a) * x + b * y
    x_list.append(x)
    y_list.append(y)
    for m in range(0, n_moms + 1):
        for n in range(0, n_moms + 1):
            xy_moms[m * (n_moms + 1) + n] += (y ** m) * (x ** n)
xy_moms = [i / num_rep for i in xy_moms]

plt.hist(x_list, bins = 40, density = True)
plt.hist(y_list, bins = 40, density = True)

x_moms = [1]
y_moms = [1]

for j in range(1, 27):
    x_moms.append(sum([k ** j for k in x_list]) / num_rep)
    y_moms.append(sum([k ** j for k in y_list]) / num_rep)


n_moms = len(x_moms[:(n_moms+1)]) - 1

n_moms = 2
moms = [float(j) for j in x_moms[:(n_moms+1)]]
work_list = [float(j) for j in x_list]

params = {}
params[1] = {}
params[1]['start'] = 100
params[1]['stop'] = 1800
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'], infty = True)


mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = True)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)


size = 1000

x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=18)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "X histogram"], prop = font1,  frameon=False, loc = 'upper left', bbox_to_anchor=(0.18, 1.03), ncols = 2)
plt.savefig(my_path + "/X_PDP_model.jpeg")




cdf_est = [0] * len(x_tmp)
work_list = sorted([float(q) for q in work_list])
cdf_true = empirical_cdf([float(q) for q in work_list], len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)


plt.plot(x_tmp, cdf_true, x_tmp, cdf_est)




n_moms = 6
moms = [float(j) for j in y_moms[:(n_moms+1)]]
work_list = [float(j) for j in y_list]

params = {}
params[1] = {}
params[1]['start'] = 8
params[1]['stop'] = 80
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'], infty = True)


mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = True)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)


size = 1000

x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=18)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "Y histogram"], prop = font1,  frameon=False, loc = 'upper left', bbox_to_anchor=(0.18, 1.03), ncols = 2)
plt.savefig(my_path + "/Y_PDP_model.jpeg")




cdf_est = [0] * len(x_tmp)
work_list = sorted([float(q) for q in work_list])
cdf_true = empirical_cdf([float(q) for q in work_list], len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)


plt.plot(x_tmp, cdf_true, x_tmp, cdf_est)




moms = xy_moms

params = {}
params[1] = {}
params[1]['start'] = 100
params[1]['stop'] = 1800
params[1]['opts'] = {'epsabs' : 1.49e-7}

params[2] = {}
params[2]['start'] = 8
params[2]['stop'] = 80
params[2]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, x_moms[1], np.sqrt(x_moms[2] - x_moms[1] ** 2),
                          params[1]['start'], params[1]['stop'])

def mes2(x1):
    return 1 / (params[2]['stop'] - params[2]['start'])


mes = dict()
mes[1] = mes1
mes[2] = mes2

X = K(params, mes, max_poly_degree=n_moms, infty = False)
X.create_orthogonal_poly_collection()
X.create_poly_combs()


vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0


tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms)


for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                              list(X.poly_combs.loc[line, :]),
                              moms)[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()


def est(*args):
    res1 = est_poly
    res2 = 1.0
    for num, j in enumerate(args):
        res2 *= mes[num+1](j)
        res1 = res1.subs(vars_[num + 1], j)
    return res1 * res2


fig = plt.figure(figsize=(10, 8))

size = 100
ax = fig.add_subplot(projection='3d')
X_tmp = np.linspace(params[1]['start'], params[1]['stop'], size)
Y_tmp = np.linspace(params[2]['start'], params[2]['stop'], size)


X_tmp, Y_tmp = np.meshgrid(X_tmp, Y_tmp)
R = np.array([est(i, j) for i, j in zip(np.ravel(X_tmp), np.ravel(Y_tmp))])
Z = R.reshape(X_tmp.shape)


surf = ax.plot_surface(X_tmp, Y_tmp, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
cbar.ax.tick_params(labelsize=15)

plt.xticks(fontsize=10, weight = 'bold')
plt.yticks(fontsize=10, weight = 'bold')
ax.set_xlabel('$X$', fontsize=20, rotation=160, weight = 'bold')
ax.set_ylabel('$Y$', fontsize=20, weight = 'bold')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('$\hat{f}$', fontsize=20, rotation = 0, weight = 'bold')
for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)
for t in ax.zaxis.get_major_ticks(): t.label.set_weight('bold')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#plt.legend(["Bivariate K-series"], prop = font1,  frameon=False)
print("Min: ", Z.min(), ", Max: ", Z.max())
plt.savefig(my_path + "/estimator_PDP.jpeg")



########################################################################################
#######################           Sampling         #####################################
########################################################################################
_size = 100
X_tmp = np.linspace(params[1]['start'], params[1]['stop'], _size)
Y_tmp = np.linspace(params[2]['start'], params[2]['stop'], _size)
Z_vals = np.zeros((_size, _size))

for i, y_val in enumerate(Y_tmp):
    for j, x_val in enumerate(X_tmp):
        Z_vals[i, j] = 20000000 * est(x_val, y_val)


Z_x_list = []
Z_y_list = []

for i in tqdm(range(Y_tmp.shape[0])):
    for j in range(X_tmp.shape[0]):
        for k in range(int(Z_vals[i, j])):
            Z_x_list.append(X_tmp[j])
            Z_y_list.append(Y_tmp[i])


plt.hist(Z_x_list, bins = 50, density = True); plt.hist(x_list, bins = 50, density = True); plt.legend(['ést', 'true',])
plt.hist(Z_y_list, bins = 50, density = True); plt.hist(y_list, bins = 50, density = True)

a = np.column_stack((Z_x_list, Z_y_list))
b = np.column_stack((x_list, y_list))

pd.DataFrame({'X_est' : a[:, 0],'Y_est' : a[:, 1]}).to_csv('PDP_estim.csv')
pd.DataFrame({ 'X_true' : b[:, 0],'Y_true' : b[:, 1]}).to_csv('PDP_true.csv')


tmp_step_x = 440
tmp_step_y = 25

t1 = np.array(np.arange(params[1]['start'], params[1]['stop'] + tmp_step_x, tmp_step_x))
t2 = np.array(np.arange(params[2]['start'], params[2]['stop'] + tmp_step_y, tmp_step_y))

step1 = tmp_step_x
step2 = tmp_step_y

num_el = {}
num_el1 = {}
num_el_list = []
num_el_list1 = []
ind = []
for j in t1:

    for i in t2:

        ind.append('[' + str(round(j, 1)) + "," + str(round(i, 1)) + '] - [' + str(round(j + step1, 1)) + "," + str(round(i, 1)) + ']\n' +
                   '[' + str(round(j, 1)) + ", " + str(round(i + step2, 1)) + '] - [' + str(round(j + step1, 1)) + ", " + str(round(i + step2, 1)) + ']'
                   )
        num_el_list.append(a[np.logical_and(np.logical_and(a[:, 0] <= j + step1, a[:, 1] <= i + step2),
                                        np.logical_and(a[:, 0] > j, a[:, 1] > i)
                                        )].shape[0])
        num_el_list1.append(b[np.logical_and(np.logical_and(b[:, 0] <= j + step1, b[:, 1] <= i + step2),
                                         np.logical_and(b[:, 0] > j, b[:, 1] > i)
                                         )].shape[0])
trsh = 500
ind = [ind[i] for i in np.where(np.array(num_el_list1) > trsh)[0]]
num_el_list = [num_el_list[i] for i in np.where(np.array(num_el_list1) > trsh)[0]]
print(sum(num_el_list))
num_el_list = [i * 100 / sum(num_el_list) for i in num_el_list]
num_el_list1 = [num_el_list1[i]  for i in np.where(np.array(num_el_list1) > trsh)[0]]
print(sum(num_el_list1))
num_el_list1 = [i * 100 / sum(num_el_list1) for i in num_el_list1]

data = pd.DataFrame({'Estimated': num_el_list, 'True' : num_el_list1}, index = ind)
data['Rectangle'] = data.index


font = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=16)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
fp = data.plot(x = 'Rectangle', y = ['Estimated', 'True'], kind = 'bar', color = ['red', 'blue'],
               width = 0.9, position = 0.5, figsize=(10,7))
plt.tight_layout(pad = 4)
fp.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
plt.legend(prop = font, loc = 'upper left',  frameon=False)
plt.ylabel('Percentage of points per region', labelpad = 5, fontsize = 15, weight = 'bold')
plt.xlabel('', labelpad = 15, fontsize = 10, weight = 'bold')
plt.xticks(rotation=55, ha='right', weight = 'bold')
plt.yticks(fontsize = 12, weight = 'bold')
plt.savefig(my_path + "/comparison_PDP_example.jpeg")




#################################################################################
############ Electricity consumption data #######################################
#################################################################################

data = pd.read_table("S1Dataset.txt", header=None)
data = data.astype('float')

data_w = data[(data[0] >= 100) & (data[0] <= 4000)]

plt.hist(data_w, density = True)
plt.hist(data_w, bins = 100)

n_moms = 7
moms = [1]

for j in range(1, n_moms + 1):
    moms.append(sum([i**j for i in data_w[0]]) / data_w[0].shape[0])


work_list = list(data_w[0])
params = {}
params[1] = {}
params[1]['start'] = 100#min(work_list)
params[1]['stop'] = 4000#max(work_list)
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return 1 / (params[1]['stop'] - params[1]['start'])
    #return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
    #                      params[1]['start'], params[1]['stop'], infty = True)
    #return trunc_Expon(x1, 1, params[1]['start'], params[1]['stop'])

mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = True)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)


kde = sm.nonparametric.KDEUnivariate(work_list)
kde.fit()  # Estimate the densities

size = 1000
#x_tmp_r = np.linspace(params[1]['start'], params[1]['stop'], size)
x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]


font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=22)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red', label = "K-series, 7 moments")
plt.hist(work_list, #bins = 30,
         density=True,  color = 'white',edgecolor='blue', linewidth=1.2, label = "Histogram")
plt.plot(kde.support, kde.density, label = "Kernel density estimator")
plt.legend(prop = font1,  frameon=False)
plt.savefig(my_path + "/K-series_real_data.jpeg")



#########################################################################################
###################  Rimless wheel walker   #############################
#########################################################################################

n_moms = 10
num_it = 2000
num_rep = 1000000
x_list = []
x1_list = []

for i in tqdm(range(num_rep)):
    p = 3.141592653589
    cos_t_2 = 0.75
    t = p * 0.1666666667
    gamma_0 = p * 0.0222222222
    st_dev = p * 0.0083333333
    variance = st_dev * st_dev
    x = Uniform(-0.1, 0.1)
    x1 = x
    for j in range(1, num_it + 1):
        w = trunc_norm(gamma_0, st_dev, gamma_0 - 0.05*p, gamma_0 + 0.05*p)
        beta1 = t/2 + w
        beta2 = t/2 - w
        update1 = -0.0542424885385433*beta1**3 + 0.526559741794786*beta1**2 - 0.00576487647751779*beta1 + 0.000467077183327398
        update2 = -0.0317914456286129*beta2**3 + 0.508956053123516*beta2**2 - 0.00110072997541027*beta2 + 4.97046341128759e-5
        x = cos_t_2 * (x + 20 * update1) - 20 * update2

        update11 = 1 - cos(beta1)
        update21 = 1 - cos(beta2)
        x1 = cos_t_2 * (x1 + 20 * update11) - 20 * update21
    x_list.append(x)
    x1_list.append(x1)

x_moms = [1]
x1_moms = [1]
for j in range(1, 27):
    x_moms.append(sum([i ** j for i in x_list]) / num_rep)
    x1_moms.append(sum([i ** j for i in x1_list]) / num_rep)


####################################################################
####################################################################
####################################################################


n_moms = 2
n_moms = len(x1_moms[:(n_moms+1)]) - 1
moms = [float(j) for j in x1_moms[:(n_moms+1)]]
work_list = [float(j) for j in x1_list]

params = {}
params[1] = {}
params[1]['start'] = -math.inf
params[1]['stop'] = math.inf
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'], infty=True)


mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty=True)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)

size = 1000
#x_tmp_r = np.linspace(params[1]['start'], params[1]['stop'], size)
x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=22)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "x histogram"], prop = font1,  frameon=False)
plt.savefig(my_path + "/x_estimator_rimless_wheel.jpeg")


cdf_est = [0] * len(x_tmp)
work_list = sorted([float(q) for q in work_list])
cdf_true = empirical_cdf(work_list, len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)


#########################################################################################
###################  2D Robotic arm model   #############################
#########################################################################################


n_moms = 2
num_it = 100
num_rep = 1000000
x_list = []
y_list = []

xy_moms = [0] * (1 + n_moms)**2

for i in tqdm(range(num_rep)):
    angles = [10, 60, 110, 160, 140, 100, 60, 20, 10, 0]
    deg_pi = 3.14159265358 * 0.005555555555555556
    sigma_2 = 0.0001
    sigma = 0.01
    x = trunc_norm(0, 0.05, -0.5, 0.5)
    y = trunc_norm(0, 0.1,-0.5,0.5)
    for j in range(1, num_it + 1):
        for an in angles:
            d = Uniform(0.98, 1.02)
            t = (an * np.pi / 180) * (1 + trunc_norm(0, 0.01,-0.05,0.05))

            x = x + d * cos(t)
            y = y + d * sin(t)
    x_list.append(x)
    y_list.append(y)
    for m in range(0, n_moms + 1):
        for n in range(0, n_moms + 1):
            xy_moms[m * (n_moms + 1) + n] += (y ** m) * (x ** n)

xy_moms = [i / num_rep for i in xy_moms]

x_moms = [1]
y_moms = [1]
for j in range(1, 27):
    x_moms.append(sum([i ** j for i in x_list]) / num_rep)
    y_moms.append(sum([i ** j for i in y_list]) / num_rep)


n_moms = 2
n_moms = len(x_moms[:(n_moms+1)]) - 1
moms = [float(j) for j in x_moms[:(n_moms+1)]]
work_list = [float(j) for j in x_list]

params = {}
params[1] = {}
params[1]['start'] = 260
params[1]['stop'] = 280
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'], infty=True)


mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = True)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)

size = 500
x_tmp_r = np.linspace(params[1]['start'], params[1]['stop'], size)
x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])


styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=18)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(np.array([266, 267, 268, 269, 270, 271]), fontsize=20, weight = 'bold', rotation = 0)
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "X histogram"], prop = font1,  frameon=False)
plt.savefig(my_path + "/X_estimator_Robotic_arm_example.jpeg")



work_list = sorted([float(q) for q in work_list])
cdf_est = [0] * len(x_tmp)
cdf_true = empirical_cdf(work_list, len(x_tmp))

for i in tqdm(range(len(x_tmp[:-1]))):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)
plt.plot(x_tmp, cdf_est, x_tmp, cdf_true)

######################################################################################

n_moms = 2
moms = [float(j) for j in y_moms[:(n_moms+1)]]
work_list = [float(j) for j in y_list]

params = {}
params[1] = {}
params[1]['start'] = 525
params[1]['stop'] = 540
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'], infty=True)


mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = True)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)

size = 500
x_tmp_r = np.linspace(params[1]['start'], params[1]['stop'], size)
x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])


styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=18)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(np.array([266, 267, 268, 269, 270, 271]), fontsize=20, weight = 'bold', rotation = 0)
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "Y histogram"], prop = font1,  frameon=False)
plt.savefig(my_path + "/Y_estimator_Robotic_arm_example.jpeg")



work_list = sorted([float(q) for q in work_list])
cdf_est = [0] * len(x_tmp)
cdf_true = empirical_cdf(work_list, len(x_tmp))

for i in tqdm(range(len(x_tmp[:-1]))):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)
plt.plot(x_tmp, cdf_est, x_tmp, cdf_true)

######################################################################################

moms = xy_moms

params = {}
params[1] = {}
params[1]['start'] = 260
params[1]['stop'] = 280
params[1]['opts'] = {'epsabs' : 1.49e-7}

params[2] = {}
params[2]['start'] = 525
params[2]['stop'] = 540
params[2]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, x_moms[1], np.sqrt(x_moms[2] - x_moms[1] ** 2),
                          params[1]['start'], params[1]['stop'])

def mes2(x1):
    return trunc_norm_pdf(x1, y_moms[1], np.sqrt(y_moms[2] - y_moms[1] ** 2),
                          params[2]['start'], params[2]['stop'])

mes = dict()
mes[1] = mes1
mes[2] = mes2

X = K(params, mes, max_poly_degree=n_moms, infty = False)
X.create_orthogonal_poly_collection()
X.create_poly_combs()


vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0


tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms)


for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                              list(X.poly_combs.loc[line, :]),
                              moms)[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()


def est(*args):
    res1 = est_poly
    res2 = 1.0
    for num, j in enumerate(args):
        res2 *= mes[num+1](j)
        res1 = res1.subs(vars_[num + 1], j)
    return res1 * res2


fig = plt.figure(figsize=(10, 8))

size = 200
ax = fig.add_subplot(projection='3d')

X_tmp = np.linspace(min(x_list),max(x_list), size)
Y_tmp = np.linspace(min(y_list), max(y_list), size)


X_tmp, Y_tmp = np.meshgrid(X_tmp, Y_tmp)
R = np.array([est(i, j) for i, j in zip(np.ravel(X_tmp), np.ravel(Y_tmp))])
Z = R.reshape(X_tmp.shape)


surf = ax.plot_surface(X_tmp, Y_tmp, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
cbar.ax.tick_params(labelsize=15)

plt.xticks(fontsize=10, weight = 'bold')
plt.yticks(fontsize=10, weight = 'bold')
ax.set_xlabel('$X$', fontsize=20, rotation=160, weight = 'bold')
ax.set_ylabel('$Y$', fontsize=20, weight = 'bold')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('$\hat{f}$', fontsize=20, rotation = 0, weight = 'bold')
for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)
for t in ax.zaxis.get_major_ticks(): t.label.set_weight('bold')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#plt.legend(["Bivariate K-series"], prop = font1,  frameon=False)
print("Min: ", Z.min(), ", Max: ", Z.max())
plt.savefig(my_path + "/estimator_robotic_arm.jpeg")


########################################################################################
#######################           Sampling         #####################################
########################################################################################
########################################################################################

_size = 100#size#int(size / np.sqrt(22))

X_tmp = np.linspace(min(x_list),max(x_list), _size)
Y_tmp = np.linspace(min(y_list), max(y_list), _size)
Z_vals = np.zeros((_size, _size))

for i, y_val in enumerate(Y_tmp):
    for j, x_val in enumerate(X_tmp):
        Z_vals[i, j] = 10000 * max(est(x_val, y_val), 0)


Z_x_list = []
Z_y_list = []

for i in tqdm(range(Y_tmp.shape[0])):
    for j in range(X_tmp.shape[0]):
        for k in range(int(Z_vals[i, j])):
            Z_x_list.append(X_tmp[j])
            Z_y_list.append(Y_tmp[i])

print(len(x_list), len(Z_x_list))
print(len(y_list), len(Z_y_list))
x_list
y_list

plt.hist(Z_x_list, bins = 50, density = True); plt.hist(x_list, bins = 50, density = True); plt.legend(['ést', 'true',])
plt.hist(Z_y_list, bins = 50, density = True); plt.hist(y_list, bins = 50, density = True); plt.legend(['ést', 'true',])

a = np.column_stack((Z_x_list, Z_y_list))
b = np.column_stack((x_list, y_list))

pd.DataFrame({'X_est' : a[:, 0],'Y_est' : a[:, 1]}).to_csv('Robotic_arm_estim.csv')
pd.DataFrame({ 'X_true' : b[:, 0],'Y_true' : b[:, 1]}).to_csv('Robotic_arm_true.csv')

tmp_step = 0.6

t1 = np.array(np.arange(min(x_list)-2, params[1]['stop'] + tmp_step, tmp_step))
t2 = np.array(np.arange(params[2]['start'], params[2]['stop'] + tmp_step, tmp_step))

step1 = tmp_step
step2 = tmp_step

num_el = {}
num_el1 = {}
num_el_list = []
num_el_list1 = []
ind = []
for j in t1:

    for i in t2:

        ind.append('[' + str(round(j, 1)) + "," + str(round(i, 1)) + '] - [' + str(round(j + step1, 1)) + "," + str(round(i, 1)) + ']\n' +
                   '[' + str(round(j, 1)) + ", " + str(round(i + step2, 1)) + '] - [' + str(round(j + step1, 1)) + ", " + str(round(i + step2, 1)) + ']'
                   )
        num_el_list.append(a[np.logical_and(np.logical_and(a[:, 0] <= j + step1, a[:, 1] <= i + step2),
                                        np.logical_and(a[:, 0] > j, a[:, 1] > i)
                                        )].shape[0])
        num_el_list1.append(b[np.logical_and(np.logical_and(b[:, 0] <= j + step1, b[:, 1] <= i + step2),
                                         np.logical_and(b[:, 0] > j, b[:, 1] > i)
                                         )].shape[0])

trsh = 10000
ind = [ind[i] for i in np.where(np.array(num_el_list1) > trsh)[0]]
num_el_list = [num_el_list[i] for i in np.where(np.array(num_el_list1) > trsh)[0]]
print(sum(num_el_list))
num_el_list = [i * 100 / sum(num_el_list) for i in num_el_list]
num_el_list1 = [num_el_list1[i]  for i in np.where(np.array(num_el_list1) > trsh)[0]]
print(sum(num_el_list1))
num_el_list1 = [i * 100 / sum(num_el_list1) for i in num_el_list1]

data = pd.DataFrame({'Estimated': num_el_list, 'True' : num_el_list1}, index = ind)
data['Rectangle'] = data.index

from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick
import matplotlib.font_manager as font_manager

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
fp = data.plot(x = 'Rectangle', y = ['Estimated', 'True'], kind = 'bar', color = ['red', 'blue'],
               width = 0.9, position = 0.5, figsize=(10,7))
plt.tight_layout(pad = 5)
fp.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
plt.legend(prop = font1, loc = 'upper left',  frameon=False)
plt.ylabel('Percentage of points per region', labelpad = 5, fontsize = 15, weight = 'bold')
plt.xlabel('', labelpad = 15, fontsize = 10, weight = 'bold')
plt.xticks(rotation=60, ha='right', weight = 'bold')
plt.yticks(fontsize = 12, weight = 'bold')
plt.savefig(my_path + "/comparison_robotic_arm.jpeg")





##################################################################################
##############§  Bivariate normal distribution  #####################
##################################################################################

n_moms = 2

a_x = -2
b_x = 2
a_y = -4
b_y = 5

def pretarget(x, y):
    mu_x = 1
    mu_y = 2

    sigma_x = 1
    sigma_y = 1
    rho = -0.3

    a_x = -2
    b_x = 2
    a_y = -4
    b_y = 5
    return trunc_Bivariate_normal_dist(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, a_x, b_x, a_y, b_y)

A = nquad(lambda t1, t2: pretarget(t1, t2),
              [[-math.inf, a_x],[-math.inf, a_y]],
              opts = {'epsabs' : 1.49e-7},
              full_output=False)[0]

B = nquad(lambda t1, t2: pretarget(t1, t2),
              [[-math.inf, b_x], [-math.inf, b_y]],
              opts={'epsabs': 1.49e-7},
              full_output=False)[0]

m1 = nquad(lambda t1, t2: pretarget(t1, t2),
              [[a_x, b_x],[a_y, b_y]],
              opts = {'epsabs' : 1.49e-7},
              full_output=False)[0]

def target(x, y):
    if (x < a_x) or (x > b_x) or (y < a_y) or (y > b_y):
        return 0.0

    return pretarget(x, y) / m1

xy_moms = []

for m in range(0, n_moms + 1):
    for n in range(0, n_moms + 1):
        tmp = nquad(lambda t1, t2: t1**n * t2**m * target(t1, t2),
                             [[a_x, b_x],[a_y, b_y]],
                             opts = {'epsabs' : 1.49e-7},
                             full_output=False)[0]
        xy_moms.append(tmp)
        if m == 0 and n == 1:
            m_x = tmp
        if m == 0 and n == 2:
            m2_x = tmp
        if m == 1 and n == 0:
            m_y = tmp
        if m == 2 and n == 0:
            m2_y = tmp


moms = xy_moms

params = {}
params[1] = {}
params[1]['start'] = a_x
params[1]['stop'] = b_x
params[1]['opts'] = {'epsabs' : 1.49e-7}

params[2] = {}
params[2]['start'] = a_y
params[2]['stop'] = b_y
params[2]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, m_x, np.sqrt(m2_x - m_x**2), a_x, b_x, infty=False)


def mes2(x1):
    return trunc_norm_pdf(x1, m_y, np.sqrt(m2_y - m_y ** 2), a_y, b_y, infty=False)


mes = dict()
mes[1] = mes1
mes[2] = mes2

X = K(params, mes, max_poly_degree=n_moms, infty = False)
X.create_orthogonal_poly_collection()
X.create_poly_combs()


vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0


tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms)


for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                              list(X.poly_combs.loc[line, :]),
                              moms)[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()


def est(*args):
    res1 = est_poly
    res2 = 1.0
    for num, j in enumerate(args):
        res2 *= mes[num+1](j)
        res1 = res1.subs(vars_[num + 1], j)
    return res1 * res2


fig = plt.figure(figsize=(12, 9))

size = 100
ax = fig.add_subplot(projection='3d')
X_tmp = np.linspace(params[1]['start'], params[1]['stop'], size)
Y_tmp = np.linspace(params[2]['start'], params[2]['stop'], size)
X_tmp, Y_tmp = np.meshgrid(X_tmp, Y_tmp)


R = np.array([est(i, j) for i, j in zip(np.ravel(X_tmp), np.ravel(Y_tmp))])
Z = R.reshape(X_tmp.shape)

surf = ax.plot_surface(X_tmp, Y_tmp, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
print("Min: ", Z.min(), ", Max: ", Z.max())
fig.savefig(my_path + "/estimator_multivar_example.jpeg")


fig = plt.figure(figsize=(12, 9))

size = 100
ax = fig.add_subplot(projection='3d')
X_tmp = np.linspace(params[1]['start'], params[1]['stop'], size)
Y_tmp = np.linspace(params[2]['start'], params[2]['stop'], size)
X_tmp, Y_tmp = np.meshgrid(X_tmp, Y_tmp)
R = [target(X_tmp[i % X_tmp.shape[0]][i // X_tmp.shape[0]],
         Y_tmp[i % Y_tmp.shape[0]][i // Y_tmp.shape[0]]
         )
     for i in range(X_tmp.shape[0]**2)
     ]

Z = np.array(R).reshape(X_tmp.shape[0], X_tmp.shape[0])

surf = ax.plot_surface(X_tmp, Y_tmp, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(0.0, 0.2)

fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
print("Min: ", Z.min(), ", Max: ", Z.max())


########################################################################################
#######################           Sampling         #####################################
########################################################################################

X_tmp = np.linspace(params[1]['start'], params[1]['stop'], size)
Y_tmp = np.linspace(params[2]['start'], params[2]['stop'], size)
Z_vals = np.zeros((size, size))
Z1_vals = np.zeros((size, size))
for i, y_val in enumerate(Y_tmp):
    for j, x_val in enumerate(X_tmp):
        Z_vals[i, j] = 5000 * est(x_val, y_val)
        Z1_vals[i, j] = 5000 * target(x_val, y_val)

Z_x_list = []
Z_y_list = []
Z1_x_list = []
Z1_y_list = []

for i in tqdm(range(Y_tmp.shape[0])):
    for j in range(X_tmp.shape[0]):
        for k in range(int(Z_vals[i, j])):
            Z_x_list.append(X_tmp[j])
            Z_y_list.append(Y_tmp[i])
        for k in range(int(Z1_vals[i, j])):
            Z1_x_list.append(X_tmp[j])
            Z1_y_list.append(Y_tmp[i])

a = np.column_stack((Z_x_list, Z_y_list))
b = np.column_stack((Z1_x_list, Z1_y_list))

t1 = np.array(np.arange(params[1]['start'], params[1]['stop'] + 1, 1))
t2 = np.array(range(params[2]['start'], params[2]['stop'] + 3, 3))

num_el = {}
num_el1 = {}
num_el_list = []
num_el_list1 = []
ind = []
for j in t1:
    num_el[j] = {}
    num_el1[j] = {}
    for i in t2:
        num_el[j][i] = a[np.logical_and(np.logical_and(a[:, 0] <= j + 1, a[:, 1] <= i + 3),
                                        np.logical_and(a[:, 0] > j, a[:, 1] > i)
                                        )].shape[0]
        num_el1[j][i] = b[np.logical_and(np.logical_and(b[:, 0] <= j + 1, b[:, 1] <= i + 3),
                                         np.logical_and(b[:, 0] > j, b[:, 1] > i)
                                         )].shape[0]

        if i == 2 and j == -1:
            ind.append('[' + str(j) + "," + str(i) + '] - [' + str(j + 1) + "," + str(i) + ']\n' +
                       '[' + str(j) + ", " + str(i + 3) + '] - [' + str(j + 1) + ", " + str(i + 3) + ']'
                       )
        else:
            ind.append('[' + str(j) + "," + str(i) + '] - [' + str(j + 1) + "," + str(i) + ']\n' +
                       '[' + str(j) + "," + str(i + 3) + '] - [' + str(j + 1) + "," + str(i + 3) + ']'
                       )
        num_el_list.append(num_el[j][i])
        num_el_list1.append(num_el1[j][i])

ind = [ind[i] for i in np.where(np.array(num_el_list1) > 2000)[0]]
num_el_list = [num_el_list[i] for i in np.where(np.array(num_el_list1) > 2000)[0]]
print(sum(num_el_list))
num_el_list = [i * 100 / sum(num_el_list) for i in num_el_list]
num_el_list1 = [num_el_list1[i]  for i in np.where(np.array(num_el_list1) > 2000)[0]]
print(sum(num_el_list1))
num_el_list1 = [i * 100 / sum(num_el_list1) for i in num_el_list1]

data = pd.DataFrame({'Estimated': num_el_list, 'True' : num_el_list1}, index = ind)
data['Rectangle'] = data.index


font = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
fp = data.plot(x = 'Rectangle', y = ['Estimated', 'True'], kind = 'bar',
               width = 0.9, position = 0.5, figsize=(10,8))
fp.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
plt.legend(prop = font)
plt.ylabel('Percentage of points per region', labelpad = 15)
plt.xticks(rotation=30, ha='right')
plt.savefig(my_path + '/Bivar_norm_comparison1.jpeg')



#########################################################################################
###################  Taylor rule model   #############################
#########################################################################################


n_moms = 10
num_it = 20
num_rep = 1000000
i1_list = []


i_moms = [0] * (1 + n_moms)**2

for ll in tqdm(range(num_rep)):
    a_p = 0.5
    a_y = 0.5
    y = 1
    y1 = 1
    p = 0.01
    p1 = 0.01
    i = 0.02
    r = 0.015
    for j in range(1, num_it + 1):
        dp = trunc_norm(0, 0.1, -1, 1)
        dy = trunc_Exp(100, 0, 1)
        p = p1
        p1 = p + dp
        y1 = 0.01 + 1.02*y
        y  = y1 - dy
        i1 = r + p + a_p * (p - p1) + a_y * log(y/y1)
    i1_list.append(i1)

i1_moms = [1]

for j in range(1, 27):
    i1_moms.append(sum([k ** j for k in i1_list]) / num_rep)



n_moms = 6
n_moms = len(i1_moms[:(n_moms+1)]) - 1
moms = [float(j) for j in i1_moms[:(n_moms+1)]]
work_list = [float(j) for j in i1_list]

params = {}
params[1] = {}
params[1]['start'] = -30
params[1]['stop'] = 30
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'])

mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty=False)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)

size = 1000
x_tmp_r = np.linspace(params[1]['start'], params[1]['stop'], size)
x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=18)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "i histogram"], prop = font1,  frameon=False)
plt.savefig(my_path + "/i_estimator_Taylor_rule.jpeg")




cdf_est = [0] * len(x_tmp)
i1_list = sorted([float(q) for q in i1_list])
cdf_true = empirical_cdf(i1_list, len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)



#########################################################################################
#########################################################################################
###################  Jaosur et al: Differential drive robot #############################
#########################################################################################
#########################################################################################
#########################################################################################

n_moms = 6
num_it = 25
num_rep = 1000000
x_list = []
y_list = []

xy_moms = [0] * (1 + n_moms)**2

for i in tqdm(range(num_rep)):

    x = Uniform(-0.1, 0.1)
    y = Uniform(-0.1, 0.1)
    t = N(0, 0.1)
    for j in range(1, num_it + 1):
        t_r = Beta(1, 3)
        t_l = Uniform(-0.1, 0.1)
        t = t + 0.1 * (2 + t_r - t_l)
        x = x + 0.05 * (4 + t_l + t_r) * cos(t)
        y = y + 0.05 * (4 + t_l + t_r) * sin(t)

    x_list.append(x)
    y_list.append(y)
    for m in range(0, n_moms + 1):
        for n in range(0, n_moms + 1):
            xy_moms[m * (n_moms + 1) + n] += (y ** m) * (x ** n)
xy_moms = [i / num_rep for i in xy_moms]
#pd.DataFrame({'x' : x_list, 'y' : y_list}).to_csv('xy_data.csv')

x_moms = [1]
y_moms = [1]
for j in range(1, 27):
    x_moms.append(sum([i ** j for i in x_list]) / num_rep)
    y_moms.append(sum([i ** j for i in y_list]) / num_rep)

n_moms = 6
n_moms = len(y_moms[:(n_moms+1)]) - 1
moms = y_moms[:(n_moms+1)]
work_list = y_list

params = {}
params[1] = {}
params[1]['start'] = -2
params[1]['stop'] = 2
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'], infty = True)

mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = True)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)

size = 1000

x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=22)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "Y histogram"], prop = font1,  frameon=False)
plt.savefig(my_path + "/Y_estimator_title_example.jpeg")



work_list = sorted([float(q) for q in work_list])
cdf_est = [0] * len(x_tmp)
cdf_true = empirical_cdf(work_list, len(x_tmp))

for i in tqdm(range(len(x_tmp[:-1]))):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)
######################################################################################

moms = xy_moms

params = {}
params[1] = {}
params[1]['start'] = -2
params[1]['stop'] = 2
params[1]['opts'] = {'epsabs' : 1.49e-7}

params[2] = {}
params[2]['start'] = -2#min(y_list)
params[2]['stop'] = 2#max(y_list)
params[2]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return trunc_norm_pdf(x1, x_moms[1], np.sqrt(x_moms[2] - x_moms[1] ** 2),
                          params[1]['start'], params[1]['stop'])

def mes2(x1):
    return trunc_norm_pdf(x1, y_moms[1], np.sqrt(y_moms[2] - y_moms[1] ** 2),
                          params[2]['start'], params[2]['stop'])

mes = dict()
mes[1] = mes1
mes[2] = mes2

X = K(params, mes, max_poly_degree=n_moms, infty = False)
X.create_orthogonal_poly_collection()
X.create_poly_combs()


vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0


tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms)


for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                              list(X.poly_combs.loc[line, :]),
                              moms)[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()


def est(*args):
    res1 = est_poly
    res2 = 1.0
    for num, j in enumerate(args):
        res2 *= mes[num+1](j)
        res1 = res1.subs(vars_[num + 1], j)
    return res1 * res2


fig = plt.figure(figsize=(10, 8))

size = 200
ax = fig.add_subplot(projection='3d')
X_tmp = np.linspace(min(x_list), max(x_list), size)
Y_tmp = np.linspace(min(y_list), max(y_list), size)


X_tmp, Y_tmp = np.meshgrid(X_tmp, Y_tmp)
R = np.array([est(i, j) for i, j in zip(np.ravel(X_tmp), np.ravel(Y_tmp))])
Z = R.reshape(X_tmp.shape)

surf = ax.plot_surface(X_tmp, Y_tmp, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

#ax.set_zlim(0.0, 6.5)
cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
cbar.ax.tick_params(labelsize=15)

plt.xticks(fontsize=10, weight = 'bold')
plt.yticks(fontsize=10, weight = 'bold')
ax.set_xlabel('$X$', fontsize=20, rotation=160, weight = 'bold')
ax.set_ylabel('$Y$', fontsize=20, weight = 'bold')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('$\hat{f}$', fontsize=20, rotation = 0, weight = 'bold')
for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)
for t in ax.zaxis.get_major_ticks(): t.label.set_weight('bold')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
print("Min: ", Z.min(), ", Max: ", Z.max())
plt.savefig(my_path + "/estimator_title_example.jpeg")


########################################################################################
#######################           Sampling         #####################################
########################################################################################
_size = 100
X_tmp = np.linspace(min(x_list), max(x_list), _size)
Y_tmp = np.linspace(min(y_list), max(y_list), _size)
Z_vals = np.zeros((_size, _size))

for i, y_val in enumerate(Y_tmp):
    for j, x_val in enumerate(X_tmp):
        Z_vals[i, j] = 240 * est(x_val, y_val)


Z_x_list = []
Z_y_list = []

for i in tqdm(range(Y_tmp.shape[0])):
    for j in range(X_tmp.shape[0]):
        for k in range(int(Z_vals[i, j])):
            Z_x_list.append(X_tmp[j])
            Z_y_list.append(Y_tmp[i])


plt.hist(Z_x_list, bins = 50, density = True); plt.hist(x_list, bins = 50, density = True); plt.legend(['ést', 'true',])
plt.hist(Z_y_list, bins = 50, density = True); plt.hist(y_list, bins = 50, density = True)

a = np.column_stack((Z_x_list, Z_y_list))
b = np.column_stack((x_list, y_list))


tmp_step = 0.35

t1 = np.array(np.arange(params[1]['start'], params[1]['stop'] + tmp_step, tmp_step))
t2 = np.array(np.arange(params[2]['start'], params[2]['stop'] + tmp_step, tmp_step))

step1 = tmp_step
step2 = tmp_step

num_el = {}
num_el1 = {}
num_el_list = []
num_el_list1 = []
ind = []
for j in t1:

    for i in t2:

        ind.append('[' + str(round(j, 1)) + "," + str(round(i, 1)) + '] - [' + str(round(j + step1, 1)) + "," + str(round(i, 1)) + ']\n' +
                   '[' + str(round(j, 1)) + ", " + str(round(i + step2, 1)) + '] - [' + str(round(j + step1, 1)) + ", " + str(round(i + step2, 1)) + ']'
                   )
        num_el_list.append(a[np.logical_and(np.logical_and(a[:, 0] <= j + step1, a[:, 1] <= i + step2),
                                        np.logical_and(a[:, 0] > j, a[:, 1] > i)
                                        )].shape[0])
        num_el_list1.append(b[np.logical_and(np.logical_and(b[:, 0] <= j + step1, b[:, 1] <= i + step2),
                                         np.logical_and(b[:, 0] > j, b[:, 1] > i)
                                         )].shape[0])

ind = [ind[i] for i in np.where(np.array(num_el_list1) > 10000)[0]]
num_el_list = [num_el_list[i] for i in np.where(np.array(num_el_list1) > 10000)[0]]
print(sum(num_el_list))
num_el_list = [i * 100 / sum(num_el_list) for i in num_el_list]
num_el_list1 = [num_el_list1[i]  for i in np.where(np.array(num_el_list1) > 10000)[0]]
print(sum(num_el_list1))
num_el_list1 = [i * 100 / sum(num_el_list1) for i in num_el_list1]

data = pd.DataFrame({'Estimated': num_el_list, 'True' : num_el_list1}, index = ind)
data['Rectangle'] = data.index


font = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=16)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
fp = data.plot(x = 'Rectangle', y = ['Estimated', 'True'], kind = 'bar', color = ['red', 'blue'],
               width = 0.9, position = 0.5, figsize=(10,7))
plt.tight_layout(pad = 4)
fp.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
plt.legend(prop = font, loc = 'upper left',  frameon=False)
plt.ylabel('Percentage of points per region', labelpad = 5, fontsize = 15, weight = 'bold')
plt.xlabel('', labelpad = 15, fontsize = 10, weight = 'bold')
plt.xticks(rotation=45, ha='right', weight = 'bold')
plt.yticks(fontsize = 12, weight = 'bold')
plt.savefig(my_path + "/comparison_title_example.jpeg")


#########################################################################################
###################  Turning vehicle model    #############################
#########################################################################################
#########################################################################################
#########################################################################################


n_moms = 8
num_it = 20
num_rep = 1000000
x_list = []
y_list = []

xy_moms = [0] * (1 + n_moms)**2


for i in tqdm(range(num_rep)):
    v0 = 10
    tau = 0.1
    q = -0.5
    psi = N(0,  np.sqrt(0.1))
    v = Uniform(6.5, 8.0)
    x = Uniform(-.1, .1)
    y = Uniform(-.5, -.3)
    for j in range(1, num_it + 1):
        w1 = Uniform(-0.1, 0.1)
        w2 = N(0, np.sqrt(0.1))
        x = x + tau * v * cos(psi)
        y = y + tau * v * sin(psi)
        v = v + tau * (q * (v - v0) + w1)
        psi = psi + w2
    x_list.append(x)
    y_list.append(y)
    for m in range(0, n_moms + 1):
        for n in range(0, n_moms + 1):
            xy_moms[m * (n_moms + 1) + n] += (y ** m) * (x ** n)

xy_moms = [i / num_rep for i in xy_moms]
x_moms = [1]
y_moms = [1]
for j in range(1, 27):
    x_moms.append(sum([i ** j for i in x_list]) / num_rep)
    y_moms.append(sum([i ** j for i in y_list]) / num_rep)


n_moms = 8
n_moms = len(y_moms[:(n_moms+1)]) - 1
moms = [float(j) for j in y_moms[:(n_moms+1)]]
work_list = [float(j) for j in y_list]

params = {}
params[1] = {}
params[1]['start'] = -20
params[1]['stop'] = 20
params[1]['opts'] = {'epsabs' : 1.49e-7}



def mes1(x1):
    return trunc_norm_pdf(x1, moms[1],  np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'])



mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = False)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)

size = 1000

x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=16)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "Y histogram"], prop = font1,  frameon=False, loc = 'upper left')
plt.savefig(my_path + "/Y_estimator_turning_vehicle_mod_example_lv.jpeg")



cdf_est = [0] * len(x_tmp)
work_list = sorted([float(q) for q in work_list])
cdf_true = empirical_cdf(work_list, len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)






n_moms = 8
n_moms = len(x_moms[:(n_moms+1)]) - 1
moms = [float(j) for j in x_moms[:(n_moms+1)]]
work_list = [float(j) for j in x_list]

params = {}
params[1] = {}
params[1]['start'] = -18
params[1]['stop'] = 18
params[1]['opts'] = {'epsabs' : 1.49e-7}



def mes1(x1):
    return trunc_norm_pdf(x1, moms[1],  np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'])



mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = False)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)

size = 1000

x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=16)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "X histogram"], prop = font1,  frameon=False, loc = 'upper left')
plt.savefig(my_path + "/X_estimator_turning_vehicle_mod_example_lv.jpeg")



cdf_est = [0] * len(x_tmp)
work_list = sorted([float(q) for q in work_list])
cdf_true = empirical_cdf(work_list, len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)


######################################################################################

moms = xy_moms

params = {}
params[1] = {}
params[1]['start'] = -18
params[1]['stop'] = 18
params[1]['opts'] = {'epsabs' : 1.49e-7}

params[2] = {}
params[2]['start'] = -20
params[2]['stop'] = 20
params[2]['opts'] = {'epsabs' : 1.49e-7}


def mes1(x1):
    return trunc_norm_pdf(x1, x_moms[1], np.sqrt(x_moms[2] - x_moms[1] ** 2),
                          params[1]['start'], params[1]['stop'])


def mes2(x1):
    return trunc_norm_pdf(x1, y_moms[1], np.sqrt(y_moms[2] - y_moms[1] ** 2),
                          params[2]['start'], params[2]['stop'])

mes = dict()
mes[1] = mes1
mes[2] = mes2

X = K(params, mes, max_poly_degree=n_moms, infty = False)
X.create_orthogonal_poly_collection()
X.create_poly_combs()


vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0


tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms)


for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                              list(X.poly_combs.loc[line, :]),
                              moms)[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()


def est(*args):
    res1 = est_poly
    res2 = 1.0
    for num, j in enumerate(args):
        res2 *= mes[num+1](j)
        res1 = res1.subs(vars_[num + 1], j)
    return res1 * res2


fig = plt.figure(figsize=(10, 8))

size = 100
ax = fig.add_subplot(projection='3d')

X_tmp = np.linspace(min(x_list), max(x_list), size)
Y_tmp = np.linspace(min(y_list), max(y_list), size)


X_tmp, Y_tmp = np.meshgrid(X_tmp, Y_tmp)
R = np.array([est(i, j) for i, j in zip(np.ravel(X_tmp), np.ravel(Y_tmp))])
Z = R.reshape(X_tmp.shape)


surf = ax.plot_surface(X_tmp, Y_tmp, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
cbar.ax.tick_params(labelsize=15)

plt.xticks(fontsize=10, weight = 'bold')
plt.yticks(fontsize=10, weight = 'bold')
ax.set_xlabel('$X$', fontsize=20, weight = 'bold')#, rotation=160 )
ax.set_ylabel('$Y$', fontsize=20, weight = 'bold')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('$\hat{f}$', fontsize=20, rotation = 0, weight = 'bold')
for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)
for t in ax.zaxis.get_major_ticks(): t.label.set_weight('bold')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.set_zticks([])
print("Min: ", Z.min(), ", Max: ", Z.max())
plt.savefig(my_path + "/estimator_turning_vehicle_mod_lv.jpeg")


#######################           Sampling         #####################################

_size = 100#size#int(size / np.sqrt(22))
#X_tmp = np.linspace(min(x_list)-2, params[1]['stop'], size)
X_tmp = np.linspace(min(x_list), max(x_list), size)
Y_tmp = np.linspace(min(y_list), max(y_list), size)
Z_vals = np.zeros((_size, _size))

for i, y_val in enumerate(Y_tmp):
    for j, x_val in enumerate(X_tmp):
        Z_vals[i, j] = 100000 * max(est(x_val, y_val), 0)


Z_x_list = []
Z_y_list = []

for i in tqdm(range(Y_tmp.shape[0])):
    for j in range(X_tmp.shape[0]):
        for k in range(int(Z_vals[i, j])):
            Z_x_list.append(X_tmp[j])
            Z_y_list.append(Y_tmp[i])

print(len(x_list), len(Z_x_list))
print(len(y_list), len(Z_y_list))


plt.hist(Z_x_list, bins = 50, density = True); plt.hist(x_list, bins = 50, density = True); plt.legend(['ést', 'true',])
plt.hist(Z_y_list, bins = 50, density = True); plt.hist(y_list, bins = 50, density = True); plt.legend(['ést', 'true',])

a = np.column_stack((Z_x_list, Z_y_list))
b = np.column_stack((x_list, y_list))

tmp_step = 5

t1 = np.array(np.arange(min(x_list)-2, params[1]['stop'] + tmp_step, tmp_step))
t2 = np.array(np.arange(params[2]['start'], params[2]['stop'] + tmp_step, tmp_step))

step1 = tmp_step
step2 = tmp_step

num_el = {}
num_el1 = {}
num_el_list = []
num_el_list1 = []
ind = []
for j in t1:

    for i in t2:

        ind.append('[' + str(round(j, 1)) + "," + str(round(i, 1)) + '] - [' + str(round(j + step1, 1)) + "," + str(round(i, 1)) + ']\n' +
                   '[' + str(round(j, 1)) + ", " + str(round(i + step2, 1)) + '] - [' + str(round(j + step1, 1)) + ", " + str(round(i + step2, 1)) + ']'
                   )
        num_el_list.append(a[np.logical_and(np.logical_and(a[:, 0] <= j + step1, a[:, 1] <= i + step2),
                                        np.logical_and(a[:, 0] > j, a[:, 1] > i)
                                        )].shape[0])
        num_el_list1.append(b[np.logical_and(np.logical_and(b[:, 0] <= j + step1, b[:, 1] <= i + step2),
                                         np.logical_and(b[:, 0] > j, b[:, 1] > i)
                                         )].shape[0])

trsh = 35000
ind = [ind[i] for i in np.where(np.array(num_el_list1) > trsh)[0]]
num_el_list = [num_el_list[i] for i in np.where(np.array(num_el_list1) > trsh)[0]]
print(sum(num_el_list))
num_el_list = [i * 100 / sum(num_el_list) for i in num_el_list]
num_el_list1 = [num_el_list1[i]  for i in np.where(np.array(num_el_list1) > trsh)[0]]
print(sum(num_el_list1))
num_el_list1 = [i * 100 / sum(num_el_list1) for i in num_el_list1]

data = pd.DataFrame({'Estimated': num_el_list, 'True' : num_el_list1}, index = ind)
data['Rectangle'] = data.index



font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
fp = data.plot(x = 'Rectangle', y = ['Estimated', 'True'], kind = 'bar', color = ['red', 'blue'],
               width = 0.9, position = 0.5, figsize=(10,7))
plt.tight_layout(pad = 4)
fp.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
plt.legend(prop = font1, loc = 'upper left',  frameon=False)
plt.ylabel('Percentage of points per region', labelpad = 5, fontsize = 15, weight = 'bold')
plt.xlabel('', labelpad = 15, fontsize = 10, weight = 'bold')
plt.xticks(rotation=60, ha='right', weight = 'bold')
plt.yticks(fontsize = 12, weight = 'bold')
plt.savefig(my_path + "/comparison_TVM_example_lv.jpeg")

#########################################################################################
###################  Turning vehicle model (Sriram)   #############################
#########################################################################################
#########################################################################################
#########################################################################################


n_moms = 8
num_it = 20
num_rep = 1000000
x_list = []
y_list = []

xy_moms = [0] * (1 + n_moms)**2


for i in tqdm(range(num_rep)):
    v0 = 10
    tau = 0.1
    q = -0.5
    psi = N(0,  0.01)
    v = Uniform(6.5, 8.0)
    x = Uniform(-.1, .1)
    y = Uniform(-.5, -.3)
    for j in range(1, num_it + 1):
        w1 = Uniform(-0.1, 0.1)
        w2 = N(0, 0.01)
        x = x + tau * v * cos(psi)
        y = y + tau * v * sin(psi)
        v = v + tau * (q * (v - v0) + w1)
        psi = psi + w2
    x_list.append(x)
    y_list.append(y)
    for m in range(0, n_moms + 1):
        for n in range(0, n_moms + 1):
            xy_moms[m * (n_moms + 1) + n] += (y ** m) * (x ** n)

xy_moms = [i / num_rep for i in xy_moms]
x_moms = [1]
y_moms = [1]
for j in range(1, 27):
    x_moms.append(sum([i ** j for i in x_list]) / num_rep)
    y_moms.append(sum([i ** j for i in y_list]) / num_rep)


n_moms = 8
n_moms = len(y_moms[:(n_moms+1)]) - 1
moms = [float(j) for j in y_moms[:(n_moms+1)]]
work_list = [float(j) for j in y_list]

params = {}
params[1] = {}
params[1]['start'] = -15
params[1]['stop'] = 15
params[1]['opts'] = {'epsabs' : 1.49e-7}



def mes1(x1):
    return trunc_norm_pdf(x1, moms[1],  np.sqrt(moms[2] - moms[1]**2),
                          params[1]['start'], params[1]['stop'])


mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = False)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)

size = 1000
#x_tmp_r = np.linspace(params[1]['start'], params[1]['stop'], size)
x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=16)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "Y histogram"], prop = font1,  frameon=False, loc = 'upper left')
plt.savefig(my_path + "/Y_estimator_turning_vehicle_sv.jpeg")



cdf_est = [0] * len(x_tmp)
work_list = sorted([float(q) for q in work_list])
cdf_true = empirical_cdf(work_list, len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)



n_moms = 8
n_moms = len(x_moms[:(n_moms+1)]) - 1
moms = [float(j) for j in x_moms[:(n_moms+1)]]
work_list = [float(j) for j in x_list]

params = {}
params[1] = {}
params[1]['start'] = 1
params[1]['stop'] = 18
params[1]['opts'] = {'epsabs' : 1.49e-7}



def mes1(x1):
    return trunc_norm_pdf(x1, moms[1],  2,
                          params[1]['start'], params[1]['stop'])


mes = dict()
mes[1] = mes1

X = K(params, mes, max_poly_degree=n_moms, infty = False)
X.create_orthogonal_poly_collection()
X.create_poly_combs()

vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0

tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms[:n_moms+1])

for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                list(X.poly_combs.loc[line, :]),
                                moms[:n_moms+1])[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()

def est(*args):
    return est1(*args, est_poly = est_poly, mes = mes)

size = 1000

x_tmp_l = np.linspace(min(work_list), max(work_list), size)

x_tmp = x_tmp_l
estimator = np.array([est(i) for i in x_tmp])

styles = ["-", "--", "--", "--", ":", ":"]

font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=16)

font2 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=24)

f = plt.figure(figsize = (10, 8))
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot(x_tmp, estimator, linestyle = styles[1], linewidth = 5, color = 'red')
plt.hist(work_list, bins = 100, density=True,  color = 'white',edgecolor='blue', linewidth=1.2)
plt.legend(["K-series", "X histogram"], prop = font1,  frameon=False, loc = 'upper left')
plt.savefig(my_path + "/X_estimator_turning_vehicle_sv.jpeg")



cdf_est = [0] * len(x_tmp)
work_list = sorted([float(q) for q in work_list])
cdf_true = empirical_cdf(work_list, len(x_tmp))

for i, val in enumerate(x_tmp[:-1]):
    cdf_est[i + 1] = cdf_est[i] + ((estimator[i + 1] + estimator[i]) / 2) * (x_tmp[i + 1] - x_tmp[i])
cdf_true = np.array(cdf_true)
cdf_est = np.array(cdf_est)
K_S_distance = max([np.abs(i - j) for i, j in zip(cdf_true,cdf_est)])
print(K_S_distance)

print(Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.05, len(x_tmp))[0] > K_S_distance)
print(Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0], Kolmogorov_Smirnov_CV(0.2, len(x_tmp))[0] > K_S_distance)


######################################################################################

moms = xy_moms

params = {}
params[1] = {}
params[1]['start'] = 1
params[1]['stop'] = 18
params[1]['opts'] = {'epsabs' : 1.49e-7}

params[2] = {}
params[2]['start'] = -15
params[2]['stop'] = 15
params[2]['opts'] = {'epsabs' : 1.49e-7}


def mes1(x1):
    return trunc_norm_pdf(x1, x_moms[1], 2,
                          params[1]['start'], params[1]['stop'])


def mes2(x1):
    return trunc_norm_pdf(x1, y_moms[1], np.sqrt(y_moms[2] - y_moms[1] ** 2),
                          params[2]['start'], params[2]['stop'])

mes = dict()
mes[1] = mes1
mes[2] = mes2

X = K(params, mes, max_poly_degree=n_moms, infty = False)
X.create_orthogonal_poly_collection()
X.create_poly_combs()


vars_ = [0]

for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
    vars_.append(symbols('x' + str(j)))

est_poly = 0


tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                          list(X.poly_combs.loc[1, :]),
                          moms)


for line in tqdm(X.poly_combs.index):
    tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                              list(X.poly_combs.loc[line, :]),
                              moms)[0]
    print(tmp)
    poly_set = X.poly_combs.loc[line]
    for key in params.keys():
        tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

    est_poly += tmp

est_poly = est_poly.as_poly()


def est(*args):
    res1 = est_poly
    res2 = 1.0
    for num, j in enumerate(args):
        res2 *= mes[num+1](j)
        res1 = res1.subs(vars_[num + 1], j)
    return res1 * res2


fig = plt.figure(figsize=(10, 8))

size = 100
ax = fig.add_subplot(projection='3d')

X_tmp = np.linspace(min(x_list), max(x_list), size)
Y_tmp = np.linspace(min(y_list), max(y_list), size)


X_tmp, Y_tmp = np.meshgrid(X_tmp, Y_tmp)
R = np.array([est(i, j) for i, j in zip(np.ravel(X_tmp), np.ravel(Y_tmp))])
Z = R.reshape(X_tmp.shape)



surf = ax.plot_surface(X_tmp, Y_tmp, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
cbar.ax.tick_params(labelsize=15)

plt.xticks(fontsize=10, weight = 'bold')
plt.yticks(fontsize=10, weight = 'bold')
ax.set_xlabel('$X$', fontsize=20, weight = 'bold')#, rotation=160 )
ax.set_ylabel('$Y$', fontsize=20, weight = 'bold')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('$\hat{f}$', fontsize=20, rotation = 0, weight = 'bold')
for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)
for t in ax.zaxis.get_major_ticks(): t.label.set_weight('bold')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax.set_zticks([])

print("Min: ", Z.min(), ", Max: ", Z.max())
plt.savefig(my_path + "/estimator_turning_sv.jpeg")




########################################################################################
########################################################################################
#######################           Sampling         #####################################
########################################################################################
########################################################################################
_size = 100
X_tmp = np.linspace(min(x_list), max(x_list), size)
Y_tmp = np.linspace(min(y_list), max(y_list), size)
Z_vals = np.zeros((_size, _size))

for i, y_val in enumerate(Y_tmp):
    for j, x_val in enumerate(X_tmp):
        Z_vals[i, j] = 100000 * max(est(x_val, y_val), 0)


Z_x_list = []
Z_y_list = []

for i in tqdm(range(Y_tmp.shape[0])):
    for j in range(X_tmp.shape[0]):
        for k in range(int(Z_vals[i, j])):
            Z_x_list.append(X_tmp[j])
            Z_y_list.append(Y_tmp[i])

print(len(x_list), len(Z_x_list))
print(len(y_list), len(Z_y_list))

a = np.column_stack((Z_x_list, Z_y_list))
b = np.column_stack((x_list, y_list))

tmp_step = 5

t1 = np.array(np.arange(min(x_list)-2, params[1]['stop'] + tmp_step, tmp_step))
t2 = np.array(np.arange(params[2]['start'], params[2]['stop'] + tmp_step, tmp_step))

step1 = tmp_step
step2 = tmp_step

num_el = {}
num_el1 = {}
num_el_list = []
num_el_list1 = []
ind = []
for j in t1:

    for i in t2:

        ind.append('[' + str(round(j, 1)) + "," + str(round(i, 1)) + '] - [' + str(round(j + step1, 1)) + "," + str(round(i, 1)) + ']\n' +
                   '[' + str(round(j, 1)) + ", " + str(round(i + step2, 1)) + '] - [' + str(round(j + step1, 1)) + ", " + str(round(i + step2, 1)) + ']'
                   )
        num_el_list.append(a[np.logical_and(np.logical_and(a[:, 0] <= j + step1, a[:, 1] <= i + step2),
                                        np.logical_and(a[:, 0] > j, a[:, 1] > i)
                                        )].shape[0])
        num_el_list1.append(b[np.logical_and(np.logical_and(b[:, 0] <= j + step1, b[:, 1] <= i + step2),
                                         np.logical_and(b[:, 0] > j, b[:, 1] > i)
                                         )].shape[0])

trsh = 35000
ind = [ind[i] for i in np.where(np.array(num_el_list1) > trsh)[0]]
num_el_list = [num_el_list[i] for i in np.where(np.array(num_el_list1) > trsh)[0]]
print(sum(num_el_list))
num_el_list = [i * 100 / sum(num_el_list) for i in num_el_list]
num_el_list1 = [num_el_list1[i]  for i in np.where(np.array(num_el_list1) > trsh)[0]]
print(sum(num_el_list1))
num_el_list1 = [i * 100 / sum(num_el_list1) for i in num_el_list1]

data = pd.DataFrame({'Estimated': num_el_list, 'True' : num_el_list1}, index = ind)
data['Rectangle'] = data.index


font1 = font_manager.FontProperties(#family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
fp = data.plot(x = 'Rectangle', y = ['Estimated', 'True'], kind = 'bar', color = ['red', 'blue'],
               width = 0.9, position = 0.5, figsize=(10,7))
plt.tight_layout(pad = 4)
fp.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
plt.legend(prop = font1, loc = 'upper left',  frameon=False)
plt.ylabel('Percentage of points per region', labelpad = 5, fontsize = 15, weight = 'bold')
plt.xlabel('', labelpad = 15, fontsize = 10, weight = 'bold')
plt.xticks(rotation=60, ha='right', weight = 'bold')
plt.yticks(fontsize = 12, weight = 'bold')
plt.savefig(my_path + "/comparison_TVM_example_sv.jpeg")



