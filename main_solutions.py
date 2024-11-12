from Solve_problem import solve_problem
import sys

'''
    1. Vasicek
    2. Random_Walk_1D
    3. Random_Walk_2D
    4. Taylor_rule
    5. Stuttering_P
    6. Differential_Drive_Robot
    7. Rimless_Wheel_Walker
    8. Turning_vehicle_model
    9. Turning_vehicle_model_Small_var
    10. PDP
    11. Robotic_Arm_2D           --- use easy_mode
'''

#solve_problem('Vasicek',
#              easy_mode = False
#              )




if __name__ == "__main__":
    args = sys.argv[1:]
    problem = args[0]
    if len(args) > 1:
        easy_mode = sys.argv[1:][1]
        solve_problem(problem, eval(easy_mode))
    else:
        solve_problem(problem)
