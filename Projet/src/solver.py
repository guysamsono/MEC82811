import numpy as np
import matplotlib.pyplot as plt

def solver_first_order(input_dict):

    a = input_dict['a']
    b = input_dict['b']
    c = input_dict['c']
    nx = input_dict['nx']
    ny = input_dict['ny']
    rho = input_dict['rho']
    cp = input_dict['cp']
    kappa = input_dict['k']
    f = input_dict['f']
    u = input_dict['u']
    temp_a = input_dict['temp_a']
    temp_b = input_dict['temp_b']
    q = input_dict['q']

    x = np.linspace(a, b, nx)
    y = np.linspace(a, c, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    A = np.zeros((nx*ny, nx*ny))
    rhs = np.zeros(nx*ny)

    for i in range(ny):
        for j in range(nx):
            k = i * nx + j 
            
            if j == 0:
                #application d'une condition de dirichlet
                A[k,k] = 1
                rhs[k] = temp_a
            
            elif j == nx-1:
                #application d'une condition de dirichlet
                A[k,k] = 1
                rhs[k] = temp_b

            elif i == 0:
                #application condition de neumman 
                A[k,k] = 1
                A[k, k+nx] = -1

            elif i == ny-1:
                #application condition de neumman
                A[k,k] = 1
                A[k, k-nx] = -1
                rhs[k] = dy*q

            else:
                #noeud intérieur
                A[k, k-1] = (rho*cp*u/dx + kappa/dx**2)
                A[k, k] = (-rho*cp*u/dx - 2*kappa/dx**2 - 2*kappa/dy**2)
                A[k, k+1] = kappa/dx**2
                A[k, k-nx] = kappa/dy**2
                A[k, k+nx] = kappa/dy**2 
                rhs[k] = -f
            
    T = np.linalg.solve(A, rhs)

    return T