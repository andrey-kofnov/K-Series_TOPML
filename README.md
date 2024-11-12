# K-Series_TOPML
This repository contains supplementary material (data and source code) to support the TOPML paper submission titled 'Moment-based Density Elicitation with Applications in Probabilistic Loops.'


##############  DESCRIPTION ################


1) requirements.txt - requirements-file for python to execute "main.py"
2) main_solutions.py - .py-file which runs the benchmarks from Table 1. 
3) K_series_computation.py - .py-file with all mathematical logic described in the paper
3) ort_poly2.py - python code to carry out Polynomial chaos expansion
4) utils.py - contains utility functions and definitions of distributions
5) S1Dataset.txt - dataset from  Munkhammar et al. [2017]
6) Sampling.py - file is used by "main_solutions.py" to create samples for the benchmarks
7) Problems.py - definition of the benchmarks and parameters to solve them


To solve the benchmark problem one should install the environment which corresponds to the requirements from "requirements.txt" (program was written for the Python version 3.8). 

Then type in command line:

python main_solutions.py 'Problem name'

or 

python main_solutions.py 'Problem name' 'easy_mode'

where easy_mode takes two possible values (True or False) to sample from 80.000 repetitions or from 1.000.000 repetitions.

Example:

python main_solutions.py Random_Walk_2D False


Full list of benchmarks:

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



To deal with predefined distribtuions, one can use (execute) the code from the file 
"Distributions.py".
