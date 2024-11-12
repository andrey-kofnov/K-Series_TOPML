from ort_poly2 import *
from scipy.stats import multivariate_normal as mvn
from plot_problem import *
from K_series_computation import *
import os

##################################################################################
#################  Truncated Bivariate normal distribution  ######################
##################################################################################


a_x, b_x = -2, 2
a_y, b_y = -4, 5

mu_x, mu_y = 1, 2
sigma_x, sigma_y = 1, 1
rho = -0.3


mu = np.array([mu_x, mu_y])
cov_mat = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                    [rho * sigma_x * sigma_y, sigma_y**2]]
                   )


def Bivariate_normal(x, y,
              mean = mu,
              cov = cov_mat,
              ):

    point = np.array([x, y])

    return mvn.pdf(point, mean=mean, cov=cov)


m1 = nquad(lambda t1, t2: Bivariate_normal(t1, t2),
              [[a_x, b_x],[a_y, b_y]],
              opts = {'epsabs' : 1.49e-7},
              full_output=False)[0]

def trunc_Bivariate_normal(x, y, mes = m1):
    if (x < a_x) or (x > b_x) or (y < a_y) or (y > b_y):
        return 0.0

    return Bivariate_normal(x, y) / mes



n_moms = 2

xy_moms = []

for m in range(0, n_moms + 1):
    for n in range(0, n_moms + 1):
        tmp = nquad(lambda t1, t2: t1 ** n * t2 ** m * trunc_Bivariate_normal(t1, t2),
                    [[a_x, b_x], [a_y, b_y]],
                    opts={'epsabs': 1.49e-7},
                    full_output=False)[0]
        xy_moms.append(tmp)


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
    return trunc_norm_pdf(x1, xy_moms[1], np.sqrt(xy_moms[2] - xy_moms[1]**2), a_x, b_x, infty=False)


def mes2(x1):
    return trunc_norm_pdf(x1, xy_moms[3], np.sqrt(xy_moms[6] - xy_moms[3] ** 2), a_y, b_y, infty=False)


mes = dict()
mes[1] = mes1
mes[2] = mes2


K_var = K(params, mes, max_poly_degree=n_moms, infty=False)
K_var.create_orthogonal_poly_collection()
K_var.create_poly_combs()

print('Start Radon-Nikodym derivative computation..')
est_poly, vars_ = compute_Radon_Nikodym_derivative(K_var, xy_moms, params)
print('Radon-Nikodym derivative computation is finished.')
print('--------------------')

def est(*args, est_poly=est_poly, mes=mes, vars_=vars_):
    return K_series(est_poly, mes, vars_, *args)

path = os.path.abspath('Figures')
size = 200

X_tmp = np.linspace(a_x, b_x, size)
Y_tmp = np.linspace(a_y, b_y, size)

X_tmp, Y_tmp = np.meshgrid(X_tmp, Y_tmp)

print('Estimates computation..')
estimator = np.array([est(i, j) for i, j in zip(np.ravel(X_tmp), np.ravel(Y_tmp))])
Z = estimator.reshape(X_tmp.shape)
print('Estimates computation is finished')
print('--------------------')
print('Plotting..')

problem_params = {
    'name' : "Truncated_Bivariate_normal"
}

plot_2d_problem([X_tmp, Y_tmp], Z, path, problem_params)



##################################################################################
#########################   Irwin-Hall distribution  #############################
##################################################################################

from utils import Irwin_Hall_pdf

n = 3

a_x, b_x = 0, n


def target(x, n=n):
    return Irwin_Hall_pdf(x, n=n)


n_moms = 6
moments = [1, 3/2, 5/2, 9/2, 43/5, 69/4, 3025/84]

params = {}
params[1] = {}
params[1]['start'] = a_x
params[1]['stop'] = b_x
params[1]['opts'] = {'epsabs' : 1.49e-7}


def mes1(x1, a=0, b=n):
    return 1 / (b - a)


mes = dict()
mes[1] = mes1


K_var = K(params, mes, max_poly_degree=n_moms, infty=False)
K_var.create_orthogonal_poly_collection()
K_var.create_poly_combs()

print('Start Radon-Nikodym derivative computation..')
est_poly, vars_ = compute_Radon_Nikodym_derivative(K_var, moments, params)
print('Radon-Nikodym derivative computation is finished.')
print('--------------------')

def est(*args, est_poly=est_poly, mes=mes, vars_=vars_):
    return K_series(est_poly, mes, vars_, *args)

path = os.path.abspath('Figures')
size = 1000

IrH_grid = np.linspace(a_x, b_x, size)


print('Estimates computation..')
estimator = np.array([est(i) for i in IrH_grid])
true = np.array([target(i) for i in IrH_grid])
print('Estimates computation is finished')
print('--------------------')
print('Plotting..')

problem_params = {
    'name' : "Irwin_hall",
    'variable' : "X",
    'truth_name' : "Irwin-hall(" + str(n) + ")",
}

plot_1d_dist(IrH_grid, estimator, true, path, problem_params)






##################################################################################
#####################   Truncated Exponential distribution  ######################
##################################################################################

from utils import trunc_Expon_pdf

a_x, b_x = 0, 1
_l = 1

def target(x, _l = _l, a_x=a_x, b_x = b_x):
    return trunc_Expon_pdf(x, _l = _l, _a = a_x, _b = b_x)


n_moms = 2

moments = [1]

for m in range(1, n_moms + 1):
    tmp = nquad(lambda t1: t1 ** m * target(t1),
                [[a_x, b_x]],
                opts={'epsabs': 1.49e-7},
                full_output=False)[0]
    moments.append(tmp)

params = {}
params[1] = {}
params[1]['start'] = a_x
params[1]['stop'] = b_x
params[1]['opts'] = {'epsabs' : 1.49e-7}


def mes1(x1, a=a_x, b=b_x):
    return 1 / (b - a)


mes = dict()
mes[1] = mes1


K_var = K(params, mes, max_poly_degree=n_moms, infty=False)
K_var.create_orthogonal_poly_collection()
K_var.create_poly_combs()

print('Start Radon-Nikodym derivative computation..')
est_poly, vars_ = compute_Radon_Nikodym_derivative(K_var, moments, params)
print('Radon-Nikodym derivative computation is finished.')
print('--------------------')

def est(*args, est_poly=est_poly, mes=mes, vars_=vars_):
    return K_series(est_poly, mes, vars_, *args)

path = os.path.abspath('Figures')
size = 1000

trunc_exp_grid = np.linspace(a_x, b_x, size)


print('Estimates computation..')
estimator = np.array([est(i) for i in trunc_exp_grid])
true = np.array([target(i) for i in trunc_exp_grid])
print('Estimates computation is finished')
print('--------------------')
print('Plotting..')

problem_params = {
    'name' : "Trunc_exponential",
    'variable' : "e",
    'truth_name' : "Trunc Exp(" + str(_l) + ", [" + str(a_x) + ", " + str(b_x) + "])",
}

plot_1d_dist(trunc_exp_grid, estimator, true, path, problem_params)



##################################################################################
########################   Electricity consumption data  #########################
##################################################################################
import statsmodels.api as sm


data = pd.read_table("S1Dataset.txt", header=None)
data = data.astype('float')

data_w = data[(data[0] >= 100) & (data[0] <= 4000)]
work_list = list(data_w[0])

n_moms = 7
moments = [1]

for j in range(1, n_moms + 1):
    moments.append(sum([i**j for i in work_list]) / len(work_list))


params = {}
params[1] = {}
params[1]['start'] = 100
params[1]['stop'] = 4000
params[1]['opts'] = {'epsabs' : 1.49e-7}

def mes1(x1):
    return 1 / (params[1]['stop'] - params[1]['start'])


mes = dict()
mes[1] = mes1

K_var = K(params, mes, max_poly_degree=n_moms, infty = True)
K_var.create_orthogonal_poly_collection()
K_var.create_poly_combs()


print('Start Radon-Nikodym derivative computation..')
est_poly, vars_ = compute_Radon_Nikodym_derivative(K_var, moments, params)
print('Radon-Nikodym derivative computation is finished.')
print('--------------------')

def est(*args, est_poly=est_poly, mes=mes, vars_=vars_):
    return K_series(est_poly, mes, vars_, *args)

path = os.path.abspath('Figures')
size = 1000

data_grid = np.linspace(min(work_list), max(work_list), size)

kde = sm.nonparametric.KDEUnivariate(work_list)
kde.fit()

print('Estimates computation..')
estimator = np.array([est(i) for i in data_grid])
print('Estimates computation is finished')
print('--------------------')
print('Plotting..')


size = 1000

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
plt.plot(data_grid, estimator, linestyle = styles[1], linewidth = 5, color = 'red', label = "K-series, 7 moments")
plt.hist(work_list, #bins = 30,
         density=True,  color = 'white',edgecolor='blue', linewidth=1.2, label = "Histogram")
plt.plot(kde.support, kde.density, label = "Kernel density estimator")
plt.legend(prop = font1,  frameon=False)
plt.savefig(path + "/K-series_real_data.jpeg")
