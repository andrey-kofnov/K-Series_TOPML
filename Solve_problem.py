import copy

from Sampling import *
from Problems import *
from K_series_computation import *
from plot_problem import plot_1d_problem, plot_2d_problem
import os


problem = "Vasicek"

def solve_problem_1d(samples, problem_params):

    moments = problem_params['moments']

    n_moms = len(moments) - 1

    X = K(problem_params['params'], problem_params['mes'], max_poly_degree=n_moms, infty=problem_params['infty'])
    X.create_orthogonal_poly_collection()
    X.create_poly_combs()

    print('Start Radon-Nikodym derivative computation..')
    est_poly, vars_ = compute_Radon_Nikodym_derivative(X, moments, problem_params['params'])
    print('Radon-Nikodym derivative computation is finished.')
    print('--------------------')

    def est(*args, est_poly = est_poly, mes = problem_params['mes'], vars_ = vars_):
        return K_series(est_poly, mes, vars_, *args)


    path = os.path.abspath('Figures')
    size = 1000
    x_tmp = np.linspace(min(samples), max(samples), size)

    print('Estimates computation..')
    estimator = np.array([est(i) for i in x_tmp])
    print('Estimates computation is finished')
    print('--------------------')
    print('Plotting..')

    plot_1d_problem(x_tmp, estimator, samples, path, problem_params)

    return




def solve_problem_2d(samples, problem_params):

    n_moms = int(np.sqrt(len(problem_params['moments']))) - 1

    X = K(problem_params['params'], problem_params['mes'], max_poly_degree=n_moms, infty=problem_params['infty'])
    X.create_orthogonal_poly_collection()
    X.create_poly_combs()

    print('Start Radon-Nikodym derivative computation..')
    est_poly, vars_ = compute_Radon_Nikodym_derivative(X, problem_params['moments'], problem_params['params'])
    print('Radon-Nikodym derivative computation is finished.')
    print('--------------------')

    def est(*args, est_poly=est_poly, mes=problem_params['mes'], vars_=vars_):
        return K_series(est_poly, mes, vars_, *args)

    path = os.path.abspath('Figures')
    size = 200

    X_tmp = np.linspace(min(samples[0]), max(samples[0]), size)
    Y_tmp = np.linspace(min(samples[1]), max(samples[1]), size)

    X_tmp, Y_tmp = np.meshgrid(X_tmp, Y_tmp)


    print('Estimates computation..')
    estimator = np.array([est(i, j) for i, j in zip(np.ravel(X_tmp), np.ravel(Y_tmp))])
    Z = estimator.reshape(X_tmp.shape)
    print('Estimates computation is finished')
    print('--------------------')
    print('Plotting..')

    plot_2d_problem([X_tmp, Y_tmp], Z, path, problem_params)



def solve_problem(problem = problem, easy_mode = False):

    num_rep = 1_000_000
    if easy_mode == True:
        num_rep = 80_000
    print('Start Sampling..')
    moments, samples = Sample_problem(problem, num_rep)
    print('Sampling is finished.')
    print('--------------------')

    problem_params = define_problem(problem, moments)

    if problem_params['n_vars'] == 1:
        solve_problem_1d(samples, problem_params)
        return

    print("Solving 1st 1D problem..")
    problem_params1 = copy.deepcopy(problem_params)
    problem_params1['moments'] = problem_params1['moments'][0]
    problem_params1['params'] = problem_params1['params'][0]
    problem_params1['mes'] = problem_params1['mes'][0]
    problem_params1['infty'] = problem_params1['infty'][0]
    problem_params1['variable'] = problem_params1['variable'][0]

    solve_problem_1d(samples[0], problem_params1)

    print("Solving 2nd 1D problem..")
    problem_params2 = copy.deepcopy(problem_params)
    problem_params2['moments'] = problem_params2['moments'][1]
    problem_params2['params'] = problem_params2['params'][1]
    problem_params2['mes'] = problem_params2['mes'][1]
    problem_params2['infty'] = problem_params2['infty'][1]
    problem_params2['variable'] = problem_params2['variable'][1]

    solve_problem_1d(samples[1], problem_params2)

    print("Solving 2D problem..")
    problem_params_2d = copy.deepcopy(problem_params)
    problem_params_2d['moments'] = problem_params_2d['moments'][2]
    problem_params_2d['params'] = problem_params_2d['params'][2]
    problem_params_2d['mes'] = problem_params_2d['mes'][2]
    problem_params_2d['infty'] = problem_params_2d['infty'][2]

    solve_problem_2d(samples, problem_params_2d)
    print("Done.")




def solve_distribution_problem(problem = problem):


    print('Start Sampling..')
    moments = Sample_problem(problem, num_rep)
    print('Sampling is finished.')
    print('--------------------')

    problem_params = define_problem(problem, moments)

    if problem_params['n_vars'] == 1:
        solve_problem_1d(samples, problem_params)
        return

    print("Solving 1st 1D problem..")
    problem_params1 = copy.deepcopy(problem_params)
    problem_params1['moments'] = problem_params1['moments'][0]
    problem_params1['params'] = problem_params1['params'][0]
    problem_params1['mes'] = problem_params1['mes'][0]
    problem_params1['infty'] = problem_params1['infty'][0]
    problem_params1['variable'] = problem_params1['variable'][0]

    solve_problem_1d(samples[0], problem_params1)

    print("Solving 2nd 1D problem..")
    problem_params2 = copy.deepcopy(problem_params)
    problem_params2['moments'] = problem_params2['moments'][1]
    problem_params2['params'] = problem_params2['params'][1]
    problem_params2['mes'] = problem_params2['mes'][1]
    problem_params2['infty'] = problem_params2['infty'][1]
    problem_params2['variable'] = problem_params2['variable'][1]

    solve_problem_1d(samples[1], problem_params2)

    print("Solving 2D problem..")
    problem_params_2d = copy.deepcopy(problem_params)
    problem_params_2d['moments'] = problem_params_2d['moments'][2]
    problem_params_2d['params'] = problem_params_2d['params'][2]
    problem_params_2d['mes'] = problem_params_2d['mes'][2]
    problem_params_2d['infty'] = problem_params_2d['infty'][2]

    solve_problem_2d(samples, problem_params_2d)
    print("Done.")

