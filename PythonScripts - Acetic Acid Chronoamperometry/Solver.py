import numpy as np
from scipy import linalg
from numpy import matmul



def LU_solver(A,b):
    P,L,U = linalg.lu(A)
    y = linalg.solve(L,matmul(P,b))
    x = linalg.solve(U,y)
    return x

def Simulation_LU_solver(A,b,x):
    P,L,U = linalg.lu(A)
    y = linalg.solve(L,matmul(P,b))
    x = linalg.solve(U,y)


if __name__ == "__main__":
    A = np.array([[17,24,1,8,15],[23,5,7,14,16],[4,6,13,20,22],[10,12,19,21,3],[11,18,25,2,9]])
    b = 65*np.ones((5,1))
    x = LU_solver(A,b)
    print(x)



