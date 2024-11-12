from ort_poly2 import trunc_norm_pdf
import math
import numpy as np


def define_problem(problem, moments):
    if problem == 'Vasicek':
        return Vasicek(moments)

    if problem == 'Random_Walk_1D':
        return random_walk_1d(moments)

    if problem == 'Random_Walk_2D':
        return random_walk_2d(moments)

    if problem == 'Taylor_rule':
        return taylor_rule_model(moments)

    if problem == 'Stuttering_P':
        return Stuttering_P(moments)

    if problem == 'Differential_Drive_Robot':
        return differential_drive_robot(moments)

    if problem == 'Rimless_Wheel_Walker':
        return rimless_wheel_walker(moments)

    if problem == 'Turning_vehicle_model':
        return turning_vehicle_model(moments)

    if problem == 'Turning_vehicle_model_Small_var':
        return turning_vehicle_model_sv(moments)

    if problem == 'PDP':
        return PDP(moments)

    if problem == 'Robotic_Arm_2D':
        return robotic_arm_2d(moments)



def Vasicek(moms):
    Vasicek = {'n_vars' : 1,
               'variable' : 'r',
               'name' : 'Vasicek',
               'discrete' : False,
               'moments' : moms,
               'infty' : True
               }

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

    Vasicek['params'] = params
    Vasicek['mes'] = mes

    return Vasicek


def Stuttering_P(moms):
    Stuttering_P = {'n_vars' : 1,
                    'variable': 'r',
                    'name': 'Stuttering_P',
                    'discrete' : False,
                    'moments' : moms,
                    'infty': True
                    }

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

    Stuttering_P['params'] = params
    Stuttering_P['mes'] = mes

    return Stuttering_P


def random_walk_1d(moms):
    random_walk_1d = {'n_vars' : 1,
                      'variable': 'X',
                      'name': 'random_walk_1d',
                      'discrete': True,
                      'moments' : moms,
                      'infty': False
                      }

    params = {}
    params[1] = {}
    params[1]['start'] = -98
    params[1]['stop'] = 102
    params[1]['opts'] = {'epsabs': 1.49e-7}

    def mes1(x1):
        return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1] ** 2),
                              params[1]['start'], params[1]['stop'], infty=True)

    mes = dict()
    mes[1] = mes1

    random_walk_1d['params'] = params
    random_walk_1d['mes'] = mes

    return random_walk_1d


def random_walk_2d(moms):
    random_walk_2d = {'n_vars' : 2,
                      'variable': ['X', 'Y'],
                      'name': 'random_walk_2d',
                      'moments' : moms,
                      'discrete': True,
                      'infty' : [False, False, False]
                      }

    random_walk_2d['params'] = []
    random_walk_2d['mes'] = []

    params1 = {}
    params1[1] = {}
    params1[1]['start'] = -100
    params1[1]['stop'] = 100
    params1[1]['opts'] = {'epsabs' : 1.49e-7}

    def mes11(x1):

        return trunc_norm_pdf(x1, moms[0][1], np.sqrt(moms[0][2] - moms[0][1]**2),
                              params1[1]['start'], params1[1]['stop'])

    mes1 = dict()
    mes1[1] = mes11

    random_walk_2d['params'].append(params1)
    random_walk_2d['mes'].append(mes1)

    params2 = {}
    params2[1] = {}
    params2[1]['start'] = -100
    params2[1]['stop'] = 100
    params2[1]['opts'] = {'epsabs': 1.49e-7}

    def mes22(x1):
        return trunc_norm_pdf(x1, moms[1][1], np.sqrt(moms[1][2] - moms[1][1] ** 2),
                              params2[1]['start'], params2[1]['stop'])

    mes2 = dict()
    mes2[1] = mes22

    random_walk_2d['params'].append(params2)
    random_walk_2d['mes'].append(mes2)

    params3 = {}
    params3[1] = {}
    params3[1]['start'] = -100
    params3[1]['stop'] = 100
    params3[1]['opts'] = {'epsabs': 1.49e-7}

    params3[2] = {}
    params3[2]['start'] = -100
    params3[2]['stop'] = 100
    params3[2]['opts'] = {'epsabs': 1.49e-7}


    def mes31(x1):
        return trunc_norm_pdf(x1, moms[0][1], np.sqrt(moms[0][2] - moms[0][1] ** 2),
                       params3[1]['start'], params3[1]['stop'])

    def mes32(x1):
        return trunc_norm_pdf(x1, moms[1][1], np.sqrt(moms[1][2] - moms[1][1] ** 2),
                              params3[2]['start'], params3[2]['stop'])

    mes3 = dict()
    mes3[1] = mes31
    mes3[2] = mes32

    random_walk_2d['params'].append(params3)
    random_walk_2d['mes'].append(mes3)


    return random_walk_2d





def taylor_rule_model(moms):
    taylor_rule_model = {'n_vars' : 1,
                         'variable': 'i',
                         'name': 'taylor_rule_model',
                         'moments' : moms,
                         'discrete': False,
                         'infty' : False
                         }

    params = {}
    params[1] = {}
    params[1]['start'] = -30
    params[1]['stop'] = 30
    params[1]['opts'] = {'epsabs': 1.49e-7}

    def mes1(x1):
        return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1] ** 2),
                              params[1]['start'], params[1]['stop'])

    mes = dict()
    mes[1] = mes1

    taylor_rule_model['params'] = params
    taylor_rule_model['mes'] = mes

    return taylor_rule_model


def differential_drive_robot(moms):
    differential_drive_robot = {'n_vars' : 2,
                                'variable': ['X', 'Y'],
                                'name': 'Differential_Drive_Robot',
                                'moments' : moms,
                                'discrete': False,
                                'infty' : [False, False, False]
                                }

    differential_drive_robot['params'] = []
    differential_drive_robot['mes'] = []

    params1 = {}
    params1[1] = {}
    params1[1]['start'] = -2
    params1[1]['stop'] = 2
    params1[1]['opts'] = {'epsabs' : 1.49e-7}

    def mes11(x1):

        return trunc_norm_pdf(x1, moms[0][1], np.sqrt(moms[0][2] - moms[0][1]**2),
                              params1[1]['start'], params1[1]['stop'])

    mes1 = dict()
    mes1[1] = mes11

    differential_drive_robot['params'].append(params1)
    differential_drive_robot['mes'].append(mes1)

    params2 = {}
    params2[1] = {}
    params2[1]['start'] = -2
    params2[1]['stop'] = 2
    params2[1]['opts'] = {'epsabs': 1.49e-7}

    def mes22(x1):
        return trunc_norm_pdf(x1, moms[1][1], np.sqrt(moms[1][2] - moms[1][1] ** 2),
                              params2[1]['start'], params2[1]['stop'])

    mes2 = dict()
    mes2[1] = mes22

    differential_drive_robot['params'].append(params2)
    differential_drive_robot['mes'].append(mes2)

    params3 = {}
    params3[1] = {}
    params3[1]['start'] = -2
    params3[1]['stop'] = 2
    params3[1]['opts'] = {'epsabs': 1.49e-7}

    params3[2] = {}
    params3[2]['start'] = -2
    params3[2]['stop'] = 2
    params3[2]['opts'] = {'epsabs': 1.49e-7}


    def mes31(x1):
        return trunc_norm_pdf(x1, moms[0][1], np.sqrt(moms[0][2] - moms[0][1] ** 2),
                       params3[1]['start'], params3[1]['stop'])

    def mes32(x1):
        return trunc_norm_pdf(x1, moms[1][1], np.sqrt(moms[1][2] - moms[1][1] ** 2),
                              params3[2]['start'], params3[2]['stop'])

    mes3 = dict()
    mes3[1] = mes31
    mes3[2] = mes32

    differential_drive_robot['params'].append(params3)
    differential_drive_robot['mes'].append(mes3)


    return differential_drive_robot




def rimless_wheel_walker(moms):
    rimless_wheel_walker = {'n_vars' : 1,
                            'variable': 'X',
                            'name': 'Differential_Drive_Robot',
                            'moments' : moms,
                            'discrete': False,
                            'infty' : True
                            }

    params = {}
    params[1] = {}
    params[1]['start'] = -math.inf
    params[1]['stop'] = math.inf
    params[1]['opts'] = {'epsabs': 1.49e-7}

    def mes1(x1):
        return trunc_norm_pdf(x1, moms[1], np.sqrt(moms[2] - moms[1] ** 2),
                              params[1]['start'], params[1]['stop'], infty=True)

    mes = dict()
    mes[1] = mes1

    rimless_wheel_walker['params'] = params
    rimless_wheel_walker['mes'] = mes

    return rimless_wheel_walker



def turning_vehicle_model(moms):
    turning_vehicle_model = {'n_vars' : 2,
                             'variable': ['X', 'Y'],
                             'name': 'Turning_vehicle_model',
                             'moments' : moms,
                             'discrete': False,
                             'infty' : [False, False, False]
                             }

    turning_vehicle_model['params'] = []
    turning_vehicle_model['mes'] = []

    params1 = {}
    params1[1] = {}
    params1[1]['start'] = -18
    params1[1]['stop'] = 18
    params1[1]['opts'] = {'epsabs' : 1.49e-7}

    def mes11(x1):

        return trunc_norm_pdf(x1, moms[0][1], np.sqrt(moms[0][2] - moms[0][1]**2),
                              params1[1]['start'], params1[1]['stop'])

    mes1 = dict()
    mes1[1] = mes11

    turning_vehicle_model['params'].append(params1)
    turning_vehicle_model['mes'].append(mes1)

    params2 = {}
    params2[1] = {}
    params2[1]['start'] = -20
    params2[1]['stop'] = 20
    params2[1]['opts'] = {'epsabs': 1.49e-7}

    def mes22(x1):
        return trunc_norm_pdf(x1, moms[1][1], np.sqrt(moms[1][2] - moms[1][1] ** 2),
                              params2[1]['start'], params2[1]['stop'])

    mes2 = dict()
    mes2[1] = mes22

    turning_vehicle_model['params'].append(params2)
    turning_vehicle_model['mes'].append(mes2)

    params3 = {}
    params3[1] = {}
    params3[1]['start'] = -18
    params3[1]['stop'] = 18
    params3[1]['opts'] = {'epsabs': 1.49e-7}

    params3[2] = {}
    params3[2]['start'] = -20
    params3[2]['stop'] = 20
    params3[2]['opts'] = {'epsabs': 1.49e-7}


    def mes31(x1):
        return trunc_norm_pdf(x1, moms[0][1], np.sqrt(moms[0][2] - moms[0][1] ** 2),
                       params3[1]['start'], params3[1]['stop'])

    def mes32(x1):
        return trunc_norm_pdf(x1, moms[1][1], np.sqrt(moms[1][2] - moms[1][1] ** 2),
                              params3[2]['start'], params3[2]['stop'])

    mes3 = dict()
    mes3[1] = mes31
    mes3[2] = mes32

    turning_vehicle_model['params'].append(params3)
    turning_vehicle_model['mes'].append(mes3)


    return turning_vehicle_model




def turning_vehicle_model_sv(moms):
    turning_vehicle_model_sv = {'n_vars' : 2,
                             'variable': ['X', 'Y'],
                             'name': 'Turning_vehicle_model_Small_var',
                             'moments' : moms,
                             'discrete': False,
                             'infty' : [False, False, False]
                             }

    turning_vehicle_model_sv['params'] = []
    turning_vehicle_model_sv['mes'] = []

    params1 = {}
    params1[1] = {}
    params1[1]['start'] = -18
    params1[1]['stop'] = 18
    params1[1]['opts'] = {'epsabs' : 1.49e-7}

    def mes11(x1):

        return trunc_norm_pdf(x1, moms[0][1], 2,
                              params1[1]['start'], params1[1]['stop'])

    mes1 = dict()
    mes1[1] = mes11

    turning_vehicle_model_sv['params'].append(params1)
    turning_vehicle_model_sv['mes'].append(mes1)

    params2 = {}
    params2[1] = {}
    params2[1]['start'] = -20
    params2[1]['stop'] = 20
    params2[1]['opts'] = {'epsabs': 1.49e-7}

    def mes22(x1):
        return trunc_norm_pdf(x1, moms[1][1], np.sqrt(moms[1][2] - moms[1][1] ** 2),
                              params2[1]['start'], params2[1]['stop'])

    mes2 = dict()
    mes2[1] = mes22

    turning_vehicle_model_sv['params'].append(params2)
    turning_vehicle_model_sv['mes'].append(mes2)

    params3 = {}
    params3[1] = {}
    params3[1]['start'] = -18
    params3[1]['stop'] = 18
    params3[1]['opts'] = {'epsabs': 1.49e-7}

    params3[2] = {}
    params3[2]['start'] = -20
    params3[2]['stop'] = 20
    params3[2]['opts'] = {'epsabs': 1.49e-7}


    def mes31(x1):
        return trunc_norm_pdf(x1, moms[0][1], 2,
                       params3[1]['start'], params3[1]['stop'])

    def mes32(x1):
        return trunc_norm_pdf(x1, moms[1][1], np.sqrt(moms[1][2] - moms[1][1] ** 2),
                              params3[2]['start'], params3[2]['stop'])

    mes3 = dict()
    mes3[1] = mes31
    mes3[2] = mes32

    turning_vehicle_model_sv['params'].append(params3)
    turning_vehicle_model_sv['mes'].append(mes3)


    return turning_vehicle_model_sv




def PDP(moms):
    PDP = {'n_vars' : 2,
           'variable': ['X', 'Y'],
           'name': 'PDP',
           'moments' : moms,
           'discrete': False,
           'infty' : [False, False, False]
           }

    PDP['params'] = []
    PDP['mes'] = []

    params1 = {}
    params1[1] = {}
    params1[1]['start'] = 100
    params1[1]['stop'] = 1800
    params1[1]['opts'] = {'epsabs' : 1.49e-7}

    def mes11(x1):

        return trunc_norm_pdf(x1, moms[0][1], np.sqrt(moms[0][2] - moms[0][1]**2),
                              params1[1]['start'], params1[1]['stop'])

    mes1 = dict()
    mes1[1] = mes11

    PDP['params'].append(params1)
    PDP['mes'].append(mes1)

    params2 = {}
    params2[1] = {}
    params2[1]['start'] = 8
    params2[1]['stop'] = 80
    params2[1]['opts'] = {'epsabs': 1.49e-7}

    def mes22(x1):
        return 1 / (params2[1]['stop'] - params2[1]['start'])

    mes2 = dict()
    mes2[1] = mes22

    PDP['params'].append(params2)
    PDP['mes'].append(mes2)

    params3 = {}
    params3[1] = {}
    params3[1]['start'] = 100
    params3[1]['stop'] = 1800
    params3[1]['opts'] = {'epsabs': 1.49e-7}

    params3[2] = {}
    params3[2]['start'] = 8
    params3[2]['stop'] = 80
    params3[2]['opts'] = {'epsabs': 1.49e-7}


    def mes31(x1):
        return trunc_norm_pdf(x1, moms[0][1], np.sqrt(moms[0][2] - moms[0][1] ** 2),
                       params3[1]['start'], params3[1]['stop'])

    def mes32(x1):
        return 1 / (params3[2]['stop'] - params3[2]['start'])

    mes3 = dict()
    mes3[1] = mes31
    mes3[2] = mes32

    PDP['params'].append(params3)
    PDP['mes'].append(mes3)


    return PDP



def robotic_arm_2d(moms):
    robotic_arm_2d = {'n_vars' : 2,
           'variable': ['X', 'Y'],
           'name': 'Robotic_Arm_2D',
           'moments' : moms,
           'discrete': False,
           'infty' : [False, False, False]
           }

    robotic_arm_2d['params'] = []
    robotic_arm_2d['mes'] = []

    params1 = {}
    params1[1] = {}
    params1[1]['start'] = 260
    params1[1]['stop'] = 280
    params1[1]['opts'] = {'epsabs' : 1.49e-7}

    def mes11(x1):

        return trunc_norm_pdf(x1, moms[0][1], np.sqrt(moms[0][2] - moms[0][1]**2),
                              params1[1]['start'], params1[1]['stop'])

    mes1 = dict()
    mes1[1] = mes11

    robotic_arm_2d['params'].append(params1)
    robotic_arm_2d['mes'].append(mes1)

    params2 = {}
    params2[1] = {}
    params2[1]['start'] = 525
    params2[1]['stop'] = 540
    params2[1]['opts'] = {'epsabs': 1.49e-7}

    def mes22(x1):
        return trunc_norm_pdf(x1, moms[1][1], np.sqrt(moms[1][2] - moms[1][1] ** 2),
                       params2[1]['start'], params2[1]['stop'])

    mes2 = dict()
    mes2[1] = mes22

    robotic_arm_2d['params'].append(params2)
    robotic_arm_2d['mes'].append(mes2)

    params3 = {}
    params3[1] = {}
    params3[1]['start'] = 260
    params3[1]['stop'] = 280
    params3[1]['opts'] = {'epsabs': 1.49e-7}

    params3[2] = {}
    params3[2]['start'] = 525
    params3[2]['stop'] = 540
    params3[2]['opts'] = {'epsabs': 1.49e-7}


    def mes31(x1):
        return trunc_norm_pdf(x1, moms[0][1], np.sqrt(moms[0][2] - moms[0][1] ** 2),
                       params3[1]['start'], params3[1]['stop'])

    def mes32(x1):
        return trunc_norm_pdf(x1, moms[1][1], np.sqrt(moms[1][2] - moms[1][1] ** 2),
                       params3[2]['start'], params3[2]['stop'])

    mes3 = dict()
    mes3[1] = mes31
    mes3[2] = mes32

    robotic_arm_2d['params'].append(params3)
    robotic_arm_2d['mes'].append(mes3)


    return robotic_arm_2d



