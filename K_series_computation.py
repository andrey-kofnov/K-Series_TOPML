import pandas as pd
from sympy import *
from ort_poly2 import *

def multivar_poly_frame(coef_mat: dict,
                        comb: list,
                        moms: list
                        ) -> (float, pd.DataFrame):
    res = pd.DataFrame({0 : [1]})
    for j in list(coef_mat.keys())[::-1]:
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

def est1(*args, vars_, est_poly, mes):
    res1 = est_poly
    res2 = 1.0
    for num, j in enumerate(args):
        res2 *= mes[num+1](j)
        res1 = res1.subs(vars_[num + 1], j)
    return res1 * res2




def compute_Radon_Nikodym_derivative(X: K, moms, params):
    vars_ = [0]

    for j in range(1, len(X.polys_coef_matrix_collection.keys()) + 1):
        vars_.append(symbols('x' + str(j)))

    est_poly = 0

    for line in tqdm(X.poly_combs.index):
        tmp = multivar_poly_frame(X.polys_coef_matrix_collection,
                                  list(X.poly_combs.loc[line, :]),
                                  moms)[0]

        tmp = np.array(tmp).item()

        poly_set = X.poly_combs.loc[line]
        for key in params.keys():
            tmp *= X.orthogonal_poly_collection[key][poly_set[key]](vars_[key])

        est_poly += tmp

    est_poly = est_poly.as_poly()

    return est_poly, vars_



def K_series(est_poly, mes, vars_, *args):
    res1 = est_poly
    res2 = 1.0
    for num, j in enumerate(args):
        res2 *= mes[num+1](j)
        res1 = res1.subs(vars_[num + 1], j)
    return res1 * res2


