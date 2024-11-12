from tqdm import tqdm
from utils import *

def sample_Vasicek(num_rep = 1_000_000,
                   num_it = 100,
                   n_moms = 2):
    r_list = []

    for _ in tqdm(range(num_rep)):
        a = 0.5
        b = 0.02
        sigma = 0.2
        r = 0.08
        w = sigma * trunc_norm(0, 1, -10, 10, size = num_it)
        for j in range(1, num_it + 1):
            r = (1 - a) * r + a * b + w[j-1]
        r_list.append(r)

    r_moms = [1]

    for j in range(1, n_moms + 1):
        r_moms.append(sum([k ** j for k in r_list]) / num_rep)

    return [float(j) for j in r_moms[:(n_moms+1)]], [float(j) for j in r_list]



def sample_Stuttering_P(num_rep = 1000000,
                        num_it = 10,
                        n_moms = 2):

    s_list = []

    for _ in tqdm(range(num_rep)):
        x = -1
        y = 1
        s = 0
        p = 0.75

        u1 = Uniform(0, 2, size = num_it)
        u2 = Uniform(0, 4, size=num_it)
        f = Bernoulli(p, size = num_it)
        for j in range(1, num_it + 1): #while true:
            x = x + f[j-1] * u1[j-1]
            y = y + f[j-1] * u2[j-1]
            s = x + y
        s_list.append(s)

    s_moms = [1]

    for j in range(1, n_moms+1):
        s_moms.append(sum([k ** j for k in s_list]) / num_rep)

    return [float(j) for j in s_moms[:(n_moms + 1)]], [float(j) for j in s_list]


def sample_1d_random_walk(num_rep = 1000000,
                          num_it = 100,
                          n_moms = 2):
    x_list = []

    for _ in tqdm(range(num_rep)):
        p = 0.5
        x = 2
        f = Bernoulli(p, size = num_it)
        for j in range(1, num_it + 1):
            c1, c2 = x - 1, x + 1
            x = c1 * f[j-1] + c2 * (1 - f[j-1])
        x_list.append(x)

    x_moms = [1]

    for j in range(1, n_moms + 1):
        x_moms.append(sum([k ** j for k in x_list]) / num_rep)

    return [float(j) for j in x_moms[:(n_moms + 1)]], [float(j) for j in x_list]


def sample_2d_random_walk(num_rep = 1000000,
                          num_it = 100,
                          n_moms = 2):


    x_list = []
    y_list = []

    xy_moms = [0] * (1 + n_moms)**2

    for _ in tqdm(range(num_rep)):
        x = 0
        y = 0
        h = Bernoulli(0.5, size = num_it)
        h1 = Bernoulli(0.5, size = num_it)
        h2 = Bernoulli(0.5, size = num_it)
        for j in range(1, num_it + 1): #while true:
            x = (x - h[j-1]) * h1[j-1] +  (x + h[j-1]) * (1 - h1[j-1])
            y = (y + (1 - h[j-1])) * h2[j-1] + (y - (1 - h[j-1])) * (1 - h2[j-1])
        x_list.append(x)
        y_list.append(y)
        for m in range(n_moms + 1):
            for n in range(n_moms + 1):
                xy_moms[m * (n_moms + 1) + n] += (y ** m) * (x ** n)
    xy_moms = [i / num_rep for i in xy_moms]

    x_moms = [1]
    y_moms = [1]

    for j in range(1, n_moms+1):
        x_moms.append(sum([k ** j for k in x_list]) / num_rep)
        y_moms.append(sum([k ** j for k in y_list]) / num_rep)

    ### RETURN: Moments of X, Moments of Y, Mixed moments of (X, Y), Sample of X, Sample of Y
    return [[float(j) for j in x_moms[:(n_moms+1)]], [float(j) for j in y_moms[:(n_moms+1)]], xy_moms], [[float(j) for j in x_list], [float(j) for j in y_list]]



def sample_PDP(num_rep=1_000_000,
               num_it = 100,
               n_moms=[2,6]):

    x_list = []
    y_list = []
    xy_moms = [0] * (1 + n_moms[0])**2

    for _ in tqdm(range(num_rep)):
        k1 = 4
        k2 = 40
        a = 0.2
        b = 4
        p = 0.5
        rho = 0.5
        y = 0
        x = 0
        c1 = Bernoulli(p, size = num_it)
        for j in range(1, num_it + 1): #while true:
            k = c1[j-1] * k1 + (1-c1[j-1]) * k2
            y, x = (1 - rho) * y + k, (1 - a) * x + b * y
        x_list.append(x)
        y_list.append(y)
        for m in range(0, n_moms[0] + 1):
            for n in range(0, n_moms[0] + 1):
                xy_moms[m * (n_moms[0] + 1) + n] += (y ** m) * (x ** n)
    xy_moms = [i / num_rep for i in xy_moms]

    x_moms = [1]
    y_moms = [1]

    for j in range(1, n_moms[0]+1):
        x_moms.append(sum([k ** j for k in x_list]) / num_rep)

    for j in range(1, n_moms[1]+1):
        y_moms.append(sum([k ** j for k in y_list]) / num_rep)

    ### RETURN: Moments of X, Moments of Y, Mixed moments of (X, Y), Sample of X, Sample of Y
    return [[float(j) for j in x_moms[:(n_moms[0] + 1)]], [float(j) for j in y_moms[:(n_moms[1] + 1)]], xy_moms], [[float(j) for j in x_list], [float(j) for j in y_list]]

def sample_rimless_wheel_walker(num_rep = 1_000_000,
                                num_it = 2000,
                                n_moms = 2):
    x_list = []

    x0 = Uniform(-0.1, 0.1, size = num_rep)
    p = np.pi
    cos_t_2 = 0.75
    t = p * 0.1666666667
    gamma_0 = p * 0.0222222222
    st_dev = p * 0.0083333333
    for i in tqdm(range(num_rep)):
        x1 = x0[i]
        w = trunc_norm(gamma_0, st_dev, gamma_0 - 0.05 * p, gamma_0 + 0.05 * p, size = num_it)
        for j in range(1, num_it + 1):
            beta1 = t / 2 + w[j-1]
            beta2 = t / 2 - w[j-1]
            update1 = 1 - cos(beta1)
            update2 = 1 - cos(beta2)
            x1 = cos_t_2 * (x1 + 20 * update1) - 20 * update2
        x_list.append(x1)

    x_moms = [1]
    for j in range(1, n_moms + 1):
        x_moms.append(sum([i ** j for i in x_list]) / num_rep)

    return [float(j) for j in x_moms[:(n_moms + 1)]], [float(j) for j in x_list]



def sample_2d_robotic_arm(num_rep = 1_000_000,
                          num_it = 100,
                          n_moms = 2):

    x_list = []
    y_list = []

    xy_moms = [0] * (1 + n_moms)**2

    x0 = trunc_norm(0, 0.05, -0.5, 0.5, size = num_rep)
    y0 = trunc_norm(0, 0.1,-0.5,0.5, size = num_rep)
    angles = [10, 60, 110, 160, 140, 100, 60, 20, 10, 0]
    for i in tqdm(range(num_rep)):
        x = x0[i]
        y = y0[i]
        for j in range(1, num_it + 1):
            for an in angles:
                d = Uniform(0.98, 1.02)
                t = (an * np.pi / 180) * (1 + trunc_norm(0, 0.01,-0.05,0.05))
                x = x + d * cos(t)
                y = y + d * sin(t)
        x_list.append(x)
        y_list.append(y)
        for m in range(n_moms + 1):
            for n in range(n_moms + 1):
                xy_moms[m * (n_moms + 1) + n] += (y ** m) * (x ** n)

    xy_moms = [i / num_rep for i in xy_moms]

    x_moms = [1]
    y_moms = [1]
    for j in range(1, n_moms + 1):
        x_moms.append(sum([i ** j for i in x_list]) / num_rep)
        y_moms.append(sum([i ** j for i in y_list]) / num_rep)

    ### RETURN: Moments of X, Moments of Y, Mixed moments of (X, Y), Sample of X, Sample of Y
    return [[float(j) for j in x_moms[:(n_moms + 1)]], [float(j) for j in y_moms[:(n_moms + 1)]], xy_moms], [[float(j) for j in x_list], [float(j) for j in y_list]]



def sample_taylor_rule_model(num_rep = 1000000,
                               num_it = 20,
                               n_moms = 6):

    i1_list = []

    for _ in tqdm(range(num_rep)):
        a_p = 0.5
        a_y = 0.5
        y = 1
        p1 = 0.01
        i1 = 0.02
        r = 0.015
        dp = trunc_norm(0, 0.1, -1, 1, size = num_it)
        dy = trunc_Exp(100, 0, 1, size = num_it)
        for j in range(1, num_it + 1):
            p = p1
            p1 = p + dp[j-1]
            y1 = 0.01 + 1.02 * y
            y = y1 - dy[j-1]
            i1 = r + p + a_p * (p - p1) + a_y * np.log(y / y1)
        i1_list.append(i1)

    i1_moms = [1]

    for j in range(1, n_moms + 1):
        i1_moms.append(sum([k ** j for k in i1_list]) / num_rep)

    return [float(j) for j in i1_moms[:(n_moms + 1)]], [float(j) for j in i1_list]


def sample_differential_drive_robot(num_rep = 1000000,
                                    num_it = 25,
                                    n_moms = 6):
    x_list = []
    y_list = []

    xy_moms = [0] * (1 + n_moms) ** 2

    x0 = Uniform(-0.1, 0.1, size = num_rep)
    y0 = Uniform(-0.1, 0.1, size = num_rep)
    t0 = N(0, 0.1, size = num_rep)
    for i in tqdm(range(num_rep)):
        x, y, t = x0[i], y0[i], t0[i]
        t_r = Beta(1, 3, size = num_it)
        t_l = Uniform(-0.1, 0.1, size = num_it)
        for j in range(1, num_it + 1):
            t = t + 0.1 * (2 + t_r[j-1] - t_l[j-1])
            x = x + 0.05 * (4 + t_l[j-1] + t_r[j-1]) * cos(t)
            y = y + 0.05 * (4 + t_l[j-1] + t_r[j-1]) * sin(t)
        x_list.append(x)
        y_list.append(y)
        for m in range(0, n_moms + 1):
            for n in range(0, n_moms + 1):
                xy_moms[m * (n_moms + 1) + n] += (y ** m) * (x ** n)
    xy_moms = [i / num_rep for i in xy_moms]

    x_moms = [1]
    y_moms = [1]
    for j in range(1, n_moms+1):
        x_moms.append(sum([i ** j for i in x_list]) / num_rep)
        y_moms.append(sum([i ** j for i in y_list]) / num_rep)

    ### RETURN: Moments of X, Moments of Y, Mixed moments of (X, Y), Sample of X, Sample of Y
    return [[float(j) for j in x_moms[:(n_moms + 1)]], [float(j) for j in y_moms[:(n_moms + 1)]], xy_moms], [[float(j) for j in x_list], [float(j) for j in y_list]]


def sample_turning_vehicle_model(num_rep = 1000000,
                                 num_it = 20,
                                 n_moms = 8):

    x_list = []
    y_list = []

    xy_moms = [0] * (1 + n_moms) ** 2

    psi0 = N(0, np.sqrt(0.1), size = num_rep)  ### Variance = sqrt(0.1)
    v0_0 = Uniform(6.5, 8.0, size = num_rep)
    x0 = Uniform(-.1, .1, size = num_rep)
    y0 = Uniform(-.5, -.3, size = num_rep)
    v0 = 10
    tau = 0.1
    q = -0.5
    for i in tqdm(range(num_rep)):
        psi, v, x, y = psi0[i], v0_0[i], x0[i], y0[i]
        w1 = Uniform(-0.1, 0.1, size = num_it)
        w2 = N(0, np.sqrt(0.1), size = num_it)
        for j in range(1, num_it + 1):
            x = x + tau * v * cos(psi)
            y = y + tau * v * sin(psi)
            v = v + tau * (q * (v - v0) + w1[j-1])
            psi = psi + w2[j-1]
        x_list.append(x)
        y_list.append(y)
        for m in range(0, n_moms + 1):
            for n in range(0, n_moms + 1):
                xy_moms[m * (n_moms + 1) + n] += (y ** m) * (x ** n)

    xy_moms = [i / num_rep for i in xy_moms]
    x_moms = [1]
    y_moms = [1]
    for j in range(1, n_moms+1):
        x_moms.append(sum([i ** j for i in x_list]) / num_rep)
        y_moms.append(sum([i ** j for i in y_list]) / num_rep)


    ### RETURN: Moments of X, Moments of Y, Mixed moments of (X, Y), Sample of X, Sample of Y
    return [[float(j) for j in x_moms[:(n_moms + 1)]], [float(j) for j in y_moms[:(n_moms + 1)]], xy_moms], [[float(j) for j in x_list], [float(j) for j in y_list]]



def sample_turning_vehicle_model_sv(num_rep = 1000000,   ## Small variance
                                    num_it = 20,
                                    n_moms = 8):

    x_list = []
    y_list = []

    xy_moms = [0] * (1 + n_moms) ** 2

    psi0 = N(0, 0.01, size = num_rep)  ### Variance = 0.01
    v0_0 = Uniform(6.5, 8.0, size = num_rep)
    x0 = Uniform(-.1, .1, size = num_rep)
    y0 = Uniform(-.5, -.3, size = num_rep)
    v0 = 10
    tau = 0.1
    q = -0.5
    for i in tqdm(range(num_rep)):
        psi, v, x, y = psi0[i], v0_0[i], x0[i], y0[i]
        w1 = Uniform(-0.1, 0.1, size = num_it)
        w2 = N(0, 0.01, size = num_it)
        for j in range(1, num_it + 1):
            x = x + tau * v * cos(psi)
            y = y + tau * v * sin(psi)
            v = v + tau * (q * (v - v0) + w1[j-1])
            psi = psi + w2[j-1]
        x_list.append(x)
        y_list.append(y)
        for m in range(0, n_moms + 1):
            for n in range(0, n_moms + 1):
                xy_moms[m * (n_moms + 1) + n] += (y ** m) * (x ** n)

    xy_moms = [i / num_rep for i in xy_moms]
    x_moms = [1]
    y_moms = [1]
    for j in range(1, n_moms+1):
        x_moms.append(sum([i ** j for i in x_list]) / num_rep)
        y_moms.append(sum([i ** j for i in y_list]) / num_rep)


    ### RETURN: Moments of X, Moments of Y, Mixed moments of (X, Y), Sample of X, Sample of Y
    return [[float(j) for j in x_moms[:(n_moms + 1)]], [float(j) for j in y_moms[:(n_moms + 1)]], xy_moms], [[float(j) for j in x_list], [float(j) for j in y_list]]


def Sample_problem(problem, num_rep=1000000):

    if problem == "Vasicek":
        return sample_Vasicek(num_rep=num_rep)

    if problem == "Stuttering_P":
        return sample_Stuttering_P(num_rep=num_rep)

    if problem == "Differential_Drive_Robot":
        return sample_differential_drive_robot(num_rep=num_rep)

    if problem == "Random_Walk_1D":
        return sample_1d_random_walk(num_rep=num_rep)

    if problem == "Random_Walk_2D":
        return sample_2d_random_walk(num_rep=num_rep)

    if problem == "Turning_vehicle_model":
        return sample_turning_vehicle_model(num_rep=num_rep)

    if problem == "Turning_vehicle_model_Small_var":
        return sample_turning_vehicle_model_sv(num_rep=num_rep)

    if problem == "Robotic_Arm_2D":
        return sample_2d_robotic_arm(num_rep=num_rep)

    if problem == "Taylor_rule":
        return sample_taylor_rule_model(num_rep=num_rep)

    if problem == "Rimless_Wheel_Walker":
        return sample_rimless_wheel_walker(num_rep=num_rep)

    if problem == "PDP":
        return sample_PDP(num_rep=num_rep)

