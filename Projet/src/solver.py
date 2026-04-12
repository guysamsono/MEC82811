import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def speed_function(c,d, y):

    '''
    Fonction qui calcule la vitesse u(y) à partir de la formule donnée dans l'énoncé.

    param c: largeur du domaine (int)
    param d: débit du fluide (float)
    param y: position en y (float)

    return: u(y) pour la position donnée (float)
    '''
    speed = (3*d)/(4*c)*(1 - (y/c)**2)
    return speed 

def compute_conservation_of_energy(T, input_dict):


    ny = input_dict['ny']
    nx = input_dict['nx']
    kappa = input_dict['k']
    rho = input_dict['rho']
    cp = input_dict['cp']
    b = input_dict['b']
    c = input_dict['c']
    d = input_dict['d']
    f = input_dict['f']

    T = np.asarray(T).reshape((ny, nx))

    dx = b / (nx - 1)
    dy = c / (ny - 1)

    total_flux_conservation = 0.0

    for j in range(ny):
        u = speed_function(c, d, j * dy)
        outward_diff = kappa * (T[j,1] - T[j,0]) / dx * dy
        outward_conv = -rho*cp*u * T[j,0] * dy
        total_flux_conservation += outward_diff + outward_conv

    for j in range(ny):
        u = speed_function(c, d, j * dy)
        outward_diff = -kappa * (T[j,-1] - T[j,-2]) / dx * dy
        outward_conv = +rho*cp*u * T[j,-1] * dy
        total_flux_conservation += outward_diff + outward_conv

    for i in range(nx):
        outward_diff = kappa * (T[1,i] - T[0,i]) / dy * dx
        total_flux_conservation += outward_diff

    for i in range(nx):
        outward_diff = -kappa * (T[-1,i] - T[-2,i]) / dy * dx
        total_flux_conservation += outward_diff

    total_flux_conservation -= f * b * c

    return total_flux_conservation



def compute_boundary_fluxes(T, input_dict):
    
    '''
    Calcule le flux de chaleur à travers la frontière supérieure du domaine.
    
    param T: tableau 1D de la température à chaque point du maillage (taille nx*ny)
    param input_dict: dictionnaire contenant les paramètres du problème (doit inclure 'nx', 'ny', 'k', 'b', 'c')

    return: flux de chaleur à travers la frontière supérieure (float)
    '''
    ny = input_dict['ny']
    nx = input_dict['nx']
    kappa = input_dict['k']
    b = input_dict['b']
    c = input_dict['c']

    T = np.asarray(T).reshape((ny, nx))

    dx = b / (nx - 1)
    dy = c / (ny - 1)

    flux_top = np.zeros(nx)

    for i in range(nx):
        dTdy_top = (3*T[-1, i] - 4*T[-2, i] + T[-3, i]) / (2*dy)
        flux_top[i] = -kappa * dTdy_top

    heat_transfer = np.trapz(flux_top, dx=dx)

    return 2*heat_transfer


def solver_first_order(input_dict, sym_test = False, source_mms = None,
                       bc_left=None, bc_right=None, bc_bottom=None, bc_top_tinf=None ):
    
    '''
    Résout l'équation de convection-diffusion en utilisant un schéma aux différences finies d'ordre 1.

    param input_dict: dictionnaire contenant les paramètres du problème (doit inclure 'nx', 'ny', 'k', 'b', 'c', 'rho', 'cp', 'd', 'f', 'temp_a', 'temp_b', 'h', 'tinf')
    param sym_test: booléen indiquant si le test de symétrie doit être effectué (True) ou non (False)
    param source_mms: fonction source supplémentaire pour le test MMS (None si non utilisé)
    param bc_left, bc_right, bc_bottom, bc_top_tinf: fonctions de condition de frontière pour les côtés gauche, droit, bas et haut (None si non utilisé)

    return: tableau 1D de la température à chaque point du maillage (taille nx*ny)
    '''

    b = input_dict['b']
    c = input_dict['c']
    d = input_dict['d']
    nx = input_dict['nx']
    ny = input_dict['ny']
    rho = input_dict['rho']
    cp = input_dict['cp']
    kappa = input_dict['k']
    f = input_dict['f']
    temp_a = input_dict['temp_a']
    temp_b = input_dict['temp_b']
    h = input_dict['h']
    tinf = input_dict['tinf']

    x = np.linspace(0, b, nx)
    if sym_test:
        y = np.linspace(-c, c, ny)
    else:
        y = np.linspace(0, c, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    N = nx * ny
    A = lil_matrix((N, N))
    rhs = np.zeros(nx*ny)

    for i in range(ny):
        for j in range(nx):
            k = i * nx + j 
            
            if j == 0:
                #application d'une condition de dirichlet
                A[k,k] = 1
                if bc_left is None:
                    rhs[k] = temp_a
                else:
                    rhs[k] = bc_left(y[i])
            
            elif j == nx-1:
                #application d'une condition de dirichlet
                A[k,k] = 1
                if bc_right is None:
                    rhs[k] = temp_b
                else:
                    rhs[k] = bc_right(y[i])

            elif sym_test and i == 0:
                #application condition de robin
                A[k,k] = -(h*dy + kappa)
                A[k, k+nx] = kappa
                rhs[k] = -dy*h*tinf

            elif not sym_test and i == 0:
                #application condition de neumman (symmétrie)
                if bc_bottom is None:
                    A[k,k] = 1
                    A[k, k+nx] = -1
                    rhs[k] = 0
                else:
                    A[k, k] = -1.0 / dy
                    A[k, k + nx] = 1.0 / dy
                    rhs[k] = bc_bottom(x[j])
                    

            elif i == ny-1:
                #application condition de robin
                A[k, k] = kappa + h*dy
                A[k, k-nx] = -kappa
                if bc_top_tinf is None:
                    rhs[k] = h*dy*tinf
                else:
                    rhs[k] = h*dy*bc_top_tinf(x[j])

            else:
                #noeud intérieur
                u = speed_function(c, d, y[i])

                A[k, k-nx] = kappa/dy**2
                A[k, k+nx] = kappa/dy**2 

                if u >= 0:
                    A[k, k-1] = (rho*cp*u/dx + kappa/dx**2)
                    A[k, k] = (-rho*cp*u/dx - 2*kappa/dx**2 - 2*kappa/dy**2)
                    A[k, k+1] = kappa/dx**2
                else:
                    A[k,k-1 ] = kappa/dx**2
                    A[k,k] =  (rho*cp*u/dx - 2*kappa/dx**2 - 2*kappa/dy**2)
                    A[k,k+1] = kappa/dx**2 - rho*cp*u/dx
                if source_mms is not None:
                    rhs[k] = -(f + source_mms(x[j], y[i]))
                else:
                    rhs[k] = -f
            
    A = A.tocsr()
    T = spsolve(A, rhs)

    return T

def solver_second_order(input_dict, scheme = 'central', sym_test = False, source_mms = None,
                        bc_left=None, bc_right=None, bc_bottom=None, bc_top_tinf=None):

    '''
    Résout l'équation de convection-diffusion en utilisant un schéma aux différences finies d'ordre 2.

    param input_dict: dictionnaire contenant les paramètres du problème (doit inclure 'nx', 'ny', 'k', 'b', 'c', 'rho', 'cp', 'd', 'f', 'temp_a', 'temp_b', 'h', 'tinf')
    param scheme: Sc
    param sym_test: booléen indiquant si le test de symétrie doit être effectué (True) ou non (False)
    param source_mms: fonction source supplémentaire pour le test MMS (None si non utilisé)
    param bc_left, bc_right, bc_bottom, bc_top_tinf: fonctions de condition de frontière pour les côtés gauche, droit, bas et haut (None si non utilisé)

    return: tableau 1D de la température à chaque point du maillage (taille nx*ny)
    '''

    b = input_dict['b']
    c = input_dict['c']
    d = input_dict['d']
    nx = input_dict['nx']
    ny = input_dict['ny']
    rho = input_dict['rho']
    cp = input_dict['cp']
    kappa = input_dict['k']
    f = input_dict['f']
    temp_a = input_dict['temp_a']
    temp_b = input_dict['temp_b']
    h = input_dict['h']
    tinf = input_dict['tinf']

    x = np.linspace(0, b, nx)
    if sym_test:
        y = np.linspace(-c, c, ny)
    else:
        y = np.linspace(0, c, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    N = nx * ny
    A = lil_matrix((N, N))
    rhs = np.zeros(nx*ny)

    if scheme == 'central':

        for i in range(ny):
            for j in range(nx):
                k = i * nx + j 
                
                if j == 0:
                    #application d'une condition de dirichlet
                    A[k,k] = 1
                    if bc_left is None:
                        rhs[k] = temp_a
                    else:
                        rhs[k] = bc_left(y[i])
                
                elif j == nx-1:
                    #application d'une condition de dirichlet
                    A[k,k] = 1
                    if bc_right is None:
                        rhs[k] = temp_b
                    else:
                        rhs[k] = bc_right(y[i])

                elif not sym_test and i == 0:
                    #application condition de neumman (symmétrie)
                    if bc_bottom is None:
                        A[k,k] = -3
                        A[k, k+nx] = 4
                        A[k, k+2*nx] = -1
                        rhs[k] = 0
                    else:
                        A[k, k] = -3.0 / (2*dy)
                        A[k, k + nx] = 4.0 / (2*dy)
                        A[k, k + 2*nx] = -1.0 / (2*dy)
                        rhs[k] = bc_bottom(x[j])
                
                elif sym_test and i == 0:
                    #application condition de robin
                    A[k, k] = -3*kappa - 2*dy*h
                    A[k, k+nx] =  4*kappa
                    A[k, k+2*nx] = -1*kappa
                    rhs[k] = -2*dy*h*tinf

                elif i == ny-1:
                    #application condition de robin
                    A[k, k] =  3*kappa + 2*dy*h
                    A[k, k-nx] = -4*kappa
                    A[k, k-2*nx] =  1*kappa
                    if bc_top_tinf is None:
                        rhs[k] =  2*dy*h*tinf
                    else:
                        rhs[k] =  2*dy*h*bc_top_tinf(x[j])

                else:
                    #noeud intérieur
                    u = speed_function(c, d, y[i])
                    A[k, k-1] = (rho*cp*u/(2*dx) + kappa/dx**2)
                    A[k, k] = (-2*kappa/dx**2 - 2*kappa/dy**2)
                    A[k, k+1] = kappa/dx**2 - rho*cp*u/(2*dx)
                    A[k, k-nx] = kappa/dy**2
                    A[k, k+nx] = kappa/dy**2 
                    if source_mms is None:
                        rhs[k] = -f
                    else:
                        rhs[k] = -(f + source_mms(x[j], y[i]))


    elif scheme == 'upwind':
        for i in range(ny):
            for j in range(nx):
                k = i * nx + j

                if j == 0:
                    #application d'une condition de dirichlet
                    A[k,k] = 1
                    if bc_left is None:
                        rhs[k] = temp_a
                    else:
                        rhs[k] = bc_left(y[i]) 

                elif j == nx-1:
                    #application d'une condition de dirichlet
                    A[k,k] = 1
                    if bc_right is None:
                        rhs[k] = temp_b
                    else:
                        rhs[k] = bc_right(y[i])
                
                elif not sym_test and i == 0:
                    #application condition de neumman (symmétrie)
                    if bc_bottom is None:
                        A[k,k] = -3
                        A[k, k+nx] = 4
                        A[k, k+2*nx] = -1
                        rhs[k] = 0
                    else:
                        A[k, k] = -3.0 / (2*dy)
                        A[k, k + nx] = 4.0 / (2*dy)
                        A[k, k + 2*nx] = -1.0 / (2*dy)
                        rhs[k] = bc_bottom(x[j])

                elif sym_test and i == 0:
                    #application condition de robin
                    A[k, k] = -3*kappa - 2*dy*h
                    A[k, k+nx] =  4*kappa
                    A[k, k+2*nx] = -1*kappa
                    rhs[k] = -2*dy*h*tinf


                elif i == ny-1:
                    #application condition de robin
                    A[k, k] =  3*kappa + 2*dy*h
                    A[k, k-nx] = -4*kappa
                    A[k, k-2*nx] =  1*kappa
                    if bc_top_tinf is None:
                        rhs[k] =  2*dy*h*tinf
                    else:
                        rhs[k] =  2*dy*h*bc_top_tinf(x[j])

                else:
                    #noeud intérieur
                    u = speed_function(c,d,y[i])
                    A[k,k-nx] = kappa/dy**2
                    A[k,k+nx] = kappa/dy**2

                    if u >= 0:
                        if j ==1:
                            #Premier point intérieur : centré ordre 2
                            A[k,k-1] = kappa/dx**2 + rho*cp*u/(2*dx)
                            A[k,k] = -2*kappa/dx**2 - 2*kappa/dy**2
                            A[k,k+1] = kappa/dx**2 - rho*cp*u/(2*dx)
                        else:
                            #Points intérieurs : upwind ordre 2
                            A[k,k-2] = -rho*cp*u/(2*dx)
                            A[k,k-1] = kappa/dx**2 + 4*rho*cp*u/(2*dx)
                            A[k,k] = -2*kappa/dx**2 - 2*kappa/dy**2 -3*rho*cp*u/(2*dx)
                            A[k,k+1] = kappa/dx**2
                    else:
                        if j == nx-2:
                            #Premier point intérieur : centré ordre 2
                            A[k,k-1] = kappa/dx**2 + rho*cp*u/(2*dx)
                            A[k,k] = -2*kappa/dx**2 - 2*kappa/dy**2
                            A[k,k+1] = kappa/dx**2 - rho*cp*u/(2*dx)
                        else:
                            #Points intérieurs : upwind ordre 2
                            A[k,k-1] = kappa/dx**2
                            A[k,k] = -2*kappa/dx**2 - 2*kappa/dy**2 + 3*rho*cp*u/(2*dx)
                            A[k,k+1] = kappa/dx**2 - 4*rho*cp*u/(2*dx)
                            A[k,k+2] = rho*cp*u/(2*dx)
                    
                    if source_mms is None:
                        rhs[k] = -f
                    else:
                        rhs[k] = -(f + source_mms(x[j], y[i]))

    A = A.tocsr()
    T = spsolve(A, rhs)

    return T

def mms_Temperature(input_dict, MMS_func):
    ny = input_dict['ny']
    nx = input_dict['nx']
    b = input_dict['b']
    c = input_dict['c']

    x = np.linspace(0,b,nx)
    y = np.linspace(0,c,ny)

    T_mms_vec = np.zeros(nx*ny)

    for i in range(ny):
        for j in range(nx):
            k = i*nx + j
            T_mms_vec[k] = MMS_func(x[j], y[i])

    return T_mms_vec

def save_as_csv(T, input_dict, filename='results/temperature_field.csv'):

    b = input_dict['b']
    c = input_dict['c']
    nx = input_dict['nx']
    ny = input_dict['ny']

    x = np.linspace(0, b, nx)
    y = np.linspace(0, c, ny)

    T = T.reshape((ny, nx))

    data = np.zeros((ny*nx, 3))
    idx = 0
    for i in range(ny):
        for j in range(nx):
            data[idx] = [x[j], y[i], T[i,j]]
            idx += 1

    np.savetxt(filename, data, delimiter=',', header='x,y,T', comments='')

    return

def save_input_as_csv(input_dict, filename='results/input_parameters.csv'):
    with open(filename, 'w') as f:
        for key, value in input_dict.items():
            f.write(f"{key},{value}\n")

    return