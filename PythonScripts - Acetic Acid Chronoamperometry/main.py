from concurrent.futures import ProcessPoolExecutor
import os
from simulation import simulation,simulation_series
import numpy as np
import itertools
np.random.seed(0)

     



if __name__ == "__main__":

    
    variables1 = np.array([5e-3]) # Scan rate, V/s
    variables2 = np.logspace(-3,-7,num=50) #keq M
    variables3 = np.logspace(0,8,num=100)#kf, s^-1
    variables4 = np.array([1e-5]) #K0， m/s. Fast enough to ensure reversibility
    variables5 = np.array([1.0]) #alpha, dimensionless 
    variables6 = np.array([[0.01,0.02,0.04,0.1]]) #bulk concentrtaion of acetic acid
    directory  = np.array(['./Series Exp T']) # where your simulation results would be stored
    commands = list(itertools.product(variables1,variables2,variables3,variables4,variables5,variables6,directory))
    commands = [(index,variables1,variables2,variables3,variables4,variables5,variables6,directory) for index,(variables1,variables2,variables3,variables4,variables5,variables6,directory) in enumerate(commands) if variables3/variables2 < 1e13 ]
    
    np.random.shuffle(commands)
    
    print(commands[0])

    
    # run in multiprocessing mode to save time. Hours to days would be necessary depends on your hardware. 
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

        executor.map(simulation_series,commands)


    
    
    
    
    

