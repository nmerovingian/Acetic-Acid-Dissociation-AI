import numpy as np
import time
from coeff import Coeff
from grid import Grid
from Solver import LU_solver
import csv
import scipy
from scipy import sparse
from scipy.sparse import linalg
import sympy
import time
import os
import math

import pandas as pd
# the total concentration of X added before any chemical equilibrium happen 

cRef=1.0 # reference concentration, 1M
P0 = 1.0 # reference pressure, 1 bar
dElectrode = 2.0*5e-6 / math.pi #radius of electrode corresponding to hemispherical electrode
KH_2 = 1292  # Henry law constant for H2
E0f = -0.2415-0.0994 # The formal potential of H+/H2 couple

Kappa = P0/ (KH_2*cRef)


DA = 1.29e-9 # diffusion coefficient of acetic acid, m^2/s , in aqueous
DB = 9.311e-9 # diffusion coefficient of H+ ion in aqueous
DY = 1.089e-9 # diffusion coefficient of acetate in aqueous
DZ = 5.11e-9 # diffusion coefficient of hydrogen in aqueous


# Scenario: fixing all diffusion coefficient to 1e-9 m^2/s 
"""
DA = 1e-9
DB = 1e-9
DY = 1e-9
DZ = 1e-9
"""

# scenario: fixing diffusion coefficient of acetic acid and acetate to be 1e-9 m^2/s
"""
DA = 1e-9
DB = 9.311e-9
DY = 1e-9
DZ = 5.11e-9
"""

# scenario: increase all diffusion coefficient by 10%
"""
DA = 1.29e-9*1.1
DB = 9.311e-9*1.1
DY = 1.089e-9 *1.1
DZ = 5.11e-9 * 1.1
"""
# scenario: decrease all diffusion coefficient by 10% 
"""
DA = 1.29e-9*0.9
DB = 9.311e-9*0.9
DY = 1.089e-9 *0.9
DZ = 5.11e-9 * 0.9
"""

def simulation_series(commands:tuple):
    index,variable1,variable2,variable3,variable4,variable5,variables6,directory = commands

    for variable6 in variables6:
        simulation((index,variable1,variable2,variable3,variable4,variable5,variable6,directory))



def simulation(commands:tuple)->None:
    index,variable1,variable2,variable3,variable4,variable5,variable6,directory = commands

    #variable4 is K0, variable5 is alpha, variable6 are reserved. 

    if not os.path.exists(directory):
        os.mkdir(directory)
    directory = f'{directory}/{index}'
    if not os.path.exists(directory):
        os.mkdir(directory)



    

    #start and end potential of voltammetric scan
    theta_i = (-1.0-E0f) * 96485/298/8.314
    theta_v = (-1.2-E0f) * 96485 /298/8.314


    #space step
    deltaX = 1e-5
    #potential step
    deltaTheta = 1e-8
    #expanding grid factor
    gamma = 0.06
    GammaT = 0.02



    # standard electrochemical rate constant. Only useful if using Butler-Volmer equation for boundary conditions
    k0 = variable4
    K0 = k0*dElectrode / DB
    alpha = 1.0

    dimScanRate = variable1
    sigma = dElectrode*dElectrode/DB*(96485/(8.314*298)) *dimScanRate


    # equilibirum constants 
    dimKeq = variable2
    
    # forward reaction rate constants
    dimKf = variable3
    # reverse reaction rate constants
    dimKb = dimKf/ dimKeq

    # convert dimensional ones to dimensionless rate constants
    Kf = dimKf * dElectrode*dElectrode/DB

    Kb = dimKb*cRef*dElectrode*dElectrode / DB

    print(f'Kf is {Kf},kb is {Kb}')

    cTstar = variable6 # Molar , mol/L

    #Get the bulk concentration of A after equilibrium
    cAstar = cTstar-(-1.0 + np.sqrt(1.0+4.0*cTstar/dimKeq))/(2.0/dimKeq)
    print(cAstar)

    concA = float(cAstar / cRef)
    concB = float(np.sqrt(cAstar*dimKeq)/cRef)
    concZ = float(np.sqrt(cAstar*dimKeq) / cRef)
    concY = 0.0

    print(f"concA {concA}, concB{concB}, concY {concY}, concZ {concZ}")

    # dimensionless diffusion coefficients of every species
    dA = DA/DB
    dB =DB/DB   # diffusion coefficient of H+
    dY =DY/DB  # diffusion coeffficient of acetate+
    dZ =DZ/DB    # diffusion coefficient of H_2 

    # the maximum number of iterations for Newton method
    number_of_iteration = 10

    deltaT = deltaTheta/sigma
    print('sigma',sigma,'DeltaT',deltaT)
    # The maximum distance of simulation
    maxT = 2.0*abs(theta_v-theta_i)/sigma
    maxX = 6.0 * np.sqrt(maxT)


    Print = False # If true, will print the concentration profiles
    printA = True
    printB = True
    printC = True
    printD = True
    printE = True 

    if not Print:
        printA = False
        printB = False
        printC = False
        printD = False
        printE = False

    pointA = -5.0
    pointB = -4.0
    pointC = -3.0
    pointD = theta_i
    pointE = theta_v


    # create the csv file to save data
    CVLocation  = f'{directory}/var1={variable1:.4E}var2={variable2:.4E}var3={variable3:.4E}var4={variable4:.4E}var5={variable5:.2E}var6={variable6:.2E}.csv'

    if os.path.exists(CVLocation):


        df = pd.read_csv(f'{CVLocation}',header=None)

        if len(df) >=870:
            print(f'{CVLocation} File exists, skipping!')
            return


    concALocation = f"concA={concA:.2E}var1={variable1:.4E}var2={variable2:.4E}var3={variable3:.4E}var4={variable4:.4E}var5={variable5:.2E}var6={variable6:.4E}.csv"
    concBLocation = f"concB={concB:.2E}var1={variable1:.4E}var2={variable2:.4E}var3={variable3:.4E}var4={variable4:.4E}var5={variable5:.2E}var6={variable6:.4E}.csv"
    concYLocation = f"concY={concY:.2E}var1={variable1:.4E}var2={variable2:.4E}var3={variable3:.4E}var4={variable4:.4E}var5={variable5:.2E}var6={variable6:.4E}.csv"
    concZLocation = f"concZ={concZ:.2E}var1={variable1:.4E}var2={variable2:.4E}var3={variable3:.4E}var4={variable4:.4E}var5={variable5:.2E}var6={variable6:.4E}.csv"

    coeff = Coeff(maxX,K0,alpha,gamma,Kappa,Kf,Kb,dA,dB,dY,dZ,concB)
    coeff.calc_n(deltaX)
    coeff.cal_time_steps(maxT,deltaT,GammaT)

    #simulation steps
    m = int(np.fabs(theta_v-theta_i)/deltaTheta)

    print(m)
    # initialzie matrix for Coeff object
    coeff.ini_jacob()
    coeff.ini_fx()
    coeff.ini_dx()
    # initialze matrix for Grid objectd
    grid = Grid(coeff.n,coeff.nT)
    grid.grid(deltaX,deltaT,gamma,GammaT)
    print('Time step',len(grid.t))
    grid.init_c(concA,concB,concY,concZ,theta_i)

    coeff.get_XX(grid.x)
    coeff.update(grid.conc,concA,concB,concY,concZ)
    coeff.Allcalc_abc(deltaT,theta_i,deltaX)
    coeff.calc_jacob(grid.conc,deltaT,theta_i)
    coeff.calc_fx(grid.conc,deltaT,theta_i)

    #print(np.linalg.det(coeff.J),coeff.J.shape,np.linalg.matrix_rank(coeff.J))





    # use spsolve for sparse matrix for acceleration
    coeff.dx = linalg.spsolve(sparse.csr_matrix(coeff.J),sparse.csr_matrix(coeff.fx[:,np.newaxis]))

    coeff.xupdate(grid.conc,theta_i)

    for i in range(number_of_iteration):
        coeff.calc_jacob(grid.conc,deltaT,theta_i)
        coeff.calc_fx(grid.conc,deltaT,theta_i)

        coeff.dx = linalg.spsolve(sparse.csr_matrix(coeff.J),sparse.csr_matrix(coeff.fx[:,np.newaxis]))
        grid.conc = coeff.xupdate(grid.conc,theta_i)
        if np.mean(np.absolute(coeff.dx)) < 1e-12:
            #print('Exit: Precision satisfied!')
            break

    if printD and math.isclose(pointD,theta_i,rel_tol=1e-3):
        grid.updateAll()
        s = f'{directory}/Point=D,Theta={theta_i:.2E}{concALocation}'
        grid.saveA(s)
        s = f'{directory}/Point=D,Theta={theta_i:.2E}{concBLocation}'
        grid.saveB(s)
        s = f'{directory}/Point=D,Theta={theta_i:.2E}{concYLocation}'
        grid.saveY(s)
        s = f'{directory}/Point=D,Theta={theta_i:.2E}{concZLocation}'
        grid.saveZ(s)

        print('Saving point D')

        printD = False

    

    f=open(CVLocation,mode='w',newline='')

    writer = csv.writer(f)

    writer.writerow([0,grid.grad()])
    Theta = theta_i
    #Theta = -7.0

    start_time = time.time()

    for i in range(2,len(grid.t)):
        deltaT = grid.t[i] - grid.t[i-1]
        


        
        if i == 3:
            print(f'Total run time is {(time.time()-start_time)*len(grid.t)/60:.2f} mins')



        coeff.update(grid.conc,concA,concB,concY,concZ)
        coeff.Allcalc_abc(deltaT,Theta,deltaX)
        for ii in range(number_of_iteration):
            coeff.calc_jacob(grid.conc,deltaT,Theta)
            coeff.calc_fx(grid.conc,deltaT,Theta)
            try:

                coeff.dx=linalg.spsolve(sparse.csr_matrix(coeff.J),sparse.csr_matrix(coeff.fx[:,np.newaxis]))
            except:
                print("Using lstsq solver! ")
                coeff.dx = np.linalg.lstsq(coeff.J,coeff.fx,rcond=None)[0]
            grid.conc = coeff.xupdate(grid.conc,Theta)

            if np.mean(np.absolute(coeff.dx)) < 1e-12:
                #print(f'Exit: Precision satisfied!\nExit at iteration {ii}')
                break
            
        if not np.isnan(grid.grad()):
            writer.writerow([grid.t[i],grid.grad()])
        else:
            print('Bad solution')

        #Save the concentration profile

        if printA and math.isclose(pointA,Theta,rel_tol=1e-3):
            grid.updateAll()
            s = f'{directory}/Point=A,Theta={Theta:.2E}{concALocation}'
            grid.saveA(s)
            s = f'{directory}/Point=A,Theta={Theta:.2E}{concBLocation}'
            grid.saveB(s)
            s = f'{directory}/Point=A,Theta={Theta:.2E}{concYLocation}'
            grid.saveY(s)
            s = f'{directory}/Point=A,Theta={Theta:.2E}{concZLocation}'
            grid.saveZ(s)

            print('Saving point A')

            printA = False
        
        if printB and math.isclose(pointB,Theta,rel_tol=1e-3):
            grid.updateAll()
            s = f'{directory}/Point=B,Theta={Theta:.2E}{concALocation}'
            grid.saveA(s)
            s = f'{directory}/Point=B,Theta={Theta:.2E}{concBLocation}'
            grid.saveB(s)
            s = f'{directory}/Point=B,Theta={Theta:.2E}{concYLocation}'
            grid.saveY(s)
            s = f'{directory}/Point=B,Theta={Theta:.2E}{concZLocation}'
            grid.saveZ(s)


            print('Saving point B')

            printB = False

        if printC and math.isclose(pointC,Theta,rel_tol=1e-3):
            grid.updateAll()
            s = f'{directory}/Point=C,Theta={Theta:.2E}{concALocation}'
            grid.saveA(s)
            s = f'{directory}/Point=C,Theta={Theta:.2E}{concBLocation}'
            grid.saveB(s)
            s = f'{directory}/Point=C,Theta={Theta:.2E}{concYLocation}'
            grid.saveY(s)
            s = f'{directory}/Point=C,Theta={Theta:.2E}{concZLocation}'
            grid.saveZ(s)

            print('Saving point C')

            printC = False

        if printE and math.isclose(pointE,Theta,rel_tol=1e-3):
            grid.updateAll()
            s = f'{directory}/Point=E,Theta={Theta:.2E}{concALocation}'
            grid.saveA(s)
            s = f'{directory}/Point=E,Theta={Theta:.2E}{concBLocation}'
            grid.saveB(s)
            s = f'{directory}/Point=E,Theta={Theta:.2E}{concYLocation}'
            grid.saveY(s)
            s = f'{directory}/Point=E,Theta={Theta:.2E}{concZLocation}'
            grid.saveZ(s)

            print('Saving point E')

            printC = False

    f.close()

    grid.updateAll()
    s = f'{directory}/Point=F,{concALocation}'
    grid.saveA(s)
    s = f'{directory}/Point=F,{concBLocation}'
    grid.saveB(s)
    s = f'{directory}/Point=F{concYLocation}'
    grid.saveY(s)
    s = f'{directory}/Point=F{concZLocation}'
    grid.saveZ(s)















if __name__ == "__main__":
    simulation_series((2, 0.005, 1.7538805018417602e-05, 3467368.504525317, 1e-05, 1.0, np.array([0.01, 0.04, 0.1 ]), './Series Experimental Exp T'))
    

    


