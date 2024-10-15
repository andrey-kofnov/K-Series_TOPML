from ort_poly2 import *


def Bivariate_normal_dist(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
    c1 = 1 / (2 * math.pi * sigma_x * sigma_y * math.sqrt(1 - rho*rho))
    c2 = -1 / (2 * (1 - rho*rho))
    m1 = ((x - mu_x)/sigma_x)**2
    m2 = ((y - mu_y) / sigma_y) ** 2
    m12 = -2*rho * ((x - mu_x)/sigma_x) * ((y - mu_y) / sigma_y)

    return c1 * math.exp(c2 * (m1 + m2 + m12))

def Bivariate_normal_cdf(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
    return float(mp.quad(lambda t1, t2: Bivariate_normal_dist(t1, t2, mu_x, mu_y, sigma_x, sigma_y, rho),
                         [-math.inf, x],
                         [-math.inf, y],
                         method='tanh-sinh'
                         )
                 )

def Bivariate_normal_mu_x(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
    return float(mp.quad(lambda t1, t2: t1 * Bivariate_normal_dist(t1, t2, mu_x, mu_y, sigma_x, sigma_y, rho),
                         [-math.inf, math.inf],
                         [-math.inf, math.inf],
                         method='tanh-sinh'
                         )
                 )

def Bivariate_normal_mu_y(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
    return float(mp.quad(lambda t1, t2: t2 * Bivariate_normal_dist(t1, t2, mu_x, mu_y, sigma_x, sigma_y, rho),
                         [-math.inf, math.inf],
                         [-math.inf, math.inf],
                         method='tanh-sinh'
                         )
                 )

def Bivariate_normal_var_x(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
    return float(mp.quad(lambda t1, t2: t1**2 * Bivariate_normal_dist(t1, t2, mu_x, mu_y, sigma_x, sigma_y, rho),
                         [-math.inf, math.inf],
                         [-math.inf, math.inf],
                         method='tanh-sinh'
                         )
                 ) - Bivariate_normal_mu_x(x, y, mu_x, mu_y, sigma_x, sigma_y, rho)**2

def Bivariate_normal_var_y(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
    return float(mp.quad(lambda t1, t2: t2**2 * Bivariate_normal_dist(t1, t2, mu_x, mu_y, sigma_x, sigma_y, rho),
                         [-math.inf, math.inf],
                         [-math.inf, math.inf],
                         method='tanh-sinh'
                         )
                 ) - Bivariate_normal_mu_y(x, y, mu_x, mu_y, sigma_x, sigma_y, rho)**2

def Bivariate_normal_cov(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
    return float(mp.quad(lambda t1, t2: t1*t2 * Bivariate_normal_dist(t1, t2, mu_x, mu_y, sigma_x, sigma_y, rho),
                         [-math.inf, math.inf],
                         [-math.inf, math.inf],
                         method='tanh-sinh'
                         )
                 ) - \
           Bivariate_normal_mu_x(x, y, mu_x, mu_y, sigma_x, sigma_y, rho) * \
           Bivariate_normal_mu_y(x, y, mu_x, mu_y, sigma_x, sigma_y, rho)

def Bivariate_normal_corr(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
    return Bivariate_normal_cov(x, y, mu_x, mu_y, sigma_x, sigma_y, rho) / math.sqrt(Bivariate_normal_var_x(x, y, mu_x, mu_y, sigma_x, sigma_y, rho) *
                                                                                     Bivariate_normal_var_y(x, y, mu_x, mu_y, sigma_x, sigma_y, rho)
                                                                                     )


from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn


def trunc_Bivariate_normal_dist(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, a_x, b_x, a_y, b_y):


    mean = np.array([mu_x, mu_y])

    cov = np.array([[sigma_x**2, rho * sigma_x*sigma_y],[rho * sigma_x*sigma_y, sigma_y**2]])

    a = np.array([a_x, a_y])
    b = np.array([b_x, b_y])


    point = np.array([x, y])


    return mvn.pdf(point, mean=mean, cov=cov)#(mvn.cdf(b, mean=mean, cov=cov) - mvn.cdf(a, mean=mean, cov=cov))