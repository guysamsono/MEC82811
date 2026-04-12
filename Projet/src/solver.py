"""
Module des solveurs numériques.

Contient les solveurs par différences finies d'ordre 1 et 2,
ainsi que les fonctions de calcul de flux et de conservation.
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def speed_function(c, d, y):
    """
    Fonction qui calcule la vitesse u(y) à partir de la formule donnée dans l'énoncé.

    :param c: largeur du domaine (int)
    :param d: débit du fluide (float)
    :param y: position en y (float)
    :return: u(y) pour la position donnée (float)
    """
    speed = (3*d)/(4*c)*(1 - (y/c)**2)
    return speed


# pylint: disable=too-many-locals
def compute_conservation_of_energy(t_array, input_dict):
    """
    Calcule le résidu de la conservation d'énergie globale sur le domaine.
    """
    ny = input_dict['ny']
    nx = input_dict['nx']
    kappa = input_dict['k']
    rho = input_dict['rho']
    cp = input_dict['cp']
    b = input_dict['b']
    c = input_dict['c']
    d = input_dict['d']
    f = input_dict['f']

    t_mesh = np.asarray(t_array).reshape((ny, nx))

    dx = b / (nx - 1)
    dy = c / (ny - 1)

    total_flux_conservation = 0.0

    for j in range(ny):
        u = speed_function(c, d, j * dy)
        outward_diff = kappa * (t_mesh[j, 1] - t_mesh[j, 0]) / dx * dy
        outward_conv = -rho*cp*u * t_mesh[j, 0] * dy
        total_flux_conservation += outward_diff + outward_conv

    for j in range(ny):
        u = speed_function(c, d, j * dy)
        outward_diff = -kappa * (t_mesh[j, -1] - t_mesh[j, -2]) / dx * dy
        outward_conv = +rho*cp*u * t_mesh[j, -1] * dy
        total_flux_conservation += outward_diff + outward_conv

    for i in range(nx):
        outward_diff = kappa * (t_mesh[1, i] - t_mesh[0, i]) / dy * dx
        total_flux_conservation += outward_diff

    for i in range(nx):
        outward_diff = -kappa * (t_mesh[-1, i] - t_mesh[-2, i]) / dy * dx
        total_flux_conservation += outward_diff

    total_flux_conservation -= f * b * c

    return total_flux_conservation


# pylint: disable=too-many-locals
def compute_boundary_fluxes(t_array, input_dict, margin_ratio=0.2):
    """
    Calcule le flux de chaleur à travers la frontière supérieure du domaine.
    Exclut les coins pour éviter les singularités.

    :param t_array: tableau 1D de la température (taille nx*ny)
    :param input_dict: dictionnaire contenant les paramètres du problème
    :param margin_ratio: fraction du domaine à ignorer de chaque côté
    :return: flux de chaleur à travers la section centrale (float)
    """
    ny = input_dict['ny']
    nx = input_dict['nx']
    kappa = input_dict['k']
    b = input_dict['b']
    c = input_dict['c']

    t_mesh = np.asarray(t_array).reshape((ny, nx))

    dx = b / (nx - 1)
    dy = c / (ny - 1)

    flux_top = np.zeros(nx)

    for i in range(nx):
        dtdy_top = (3*t_mesh[-1, i] - 4*t_mesh[-2, i] + t_mesh[-3, i]) / (2*dy)
        flux_top[i] = -kappa * dtdy_top

    x_start = b * margin_ratio
    x_end = b * (1 - margin_ratio)

    i_start = int(round(x_start / dx))
    i_end = int(round(x_end / dx))

    flux_top_center = flux_top[i_start : i_end + 1]

    heat_transfer = np.trapezoid(flux_top_center, dx=dx)

    return 2 * heat_transfer


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
# pylint: disable=too-many-branches, too-many-statements
def solver_first_order(
        input_dict, sym_test=False, source_mms=None,
        bc_left=None, bc_right=None, bc_bottom=None, bc_top_tinf=None
):
    """
    Résout l'équation de convection-diffusion (schéma ordre 1).

    :param input_dict: paramètres du problème ('nx', 'ny', 'k', 'b', etc.)
    :param sym_test: booléen (True si test de symétrie)
    :param source_mms: fonction source supplémentaire pour le test MMS
    :param bc_left, bc_right, bc_bottom, bc_top_tinf: fonctions de conditions
        aux limites (None par défaut)
    :return: tableau 1D de la température
    """
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

    n_nodes = nx * ny
    matrix_a = lil_matrix((n_nodes, n_nodes))
    rhs = np.zeros(nx*ny)

    for i in range(ny):
        for j in range(nx):
            k = i * nx + j

            if j == 0:
                # condition de dirichlet
                matrix_a[k, k] = 1
                if bc_left is None:
                    rhs[k] = temp_a
                else:
                    rhs[k] = bc_left(y[i])

            elif j == nx-1:
                # condition de dirichlet
                matrix_a[k, k] = 1
                if bc_right is None:
                    rhs[k] = temp_b
                else:
                    rhs[k] = bc_right(y[i])

            elif sym_test and i == 0:
                # condition de robin
                matrix_a[k, k] = -(h*dy + kappa)
                matrix_a[k, k+nx] = kappa
                rhs[k] = -dy*h*tinf

            elif not sym_test and i == 0:
                # condition de neumman (symmétrie)
                if bc_bottom is None:
                    matrix_a[k, k] = 1
                    matrix_a[k, k+nx] = -1
                    rhs[k] = 0
                else:
                    matrix_a[k, k] = -1.0 / dy
                    matrix_a[k, k + nx] = 1.0 / dy
                    rhs[k] = bc_bottom(x[j])

            elif i == ny-1:
                # condition de robin
                matrix_a[k, k] = kappa + h*dy
                matrix_a[k, k-nx] = -kappa
                if bc_top_tinf is None:
                    rhs[k] = h*dy*tinf
                else:
                    rhs[k] = h*dy*bc_top_tinf(x[j])

            else:
                # noeud intérieur
                u = speed_function(c, d, y[i])

                matrix_a[k, k-nx] = kappa/dy**2
                matrix_a[k, k+nx] = kappa/dy**2

                if u >= 0:
                    matrix_a[k, k-1] = rho*cp*u/dx + kappa/dx**2
                    matrix_a[k, k] = -rho*cp*u/dx - 2*kappa/dx**2 - 2*kappa/dy**2
                    matrix_a[k, k+1] = kappa/dx**2
                else:
                    matrix_a[k, k-1] = kappa/dx**2
                    matrix_a[k, k] = rho*cp*u/dx - 2*kappa/dx**2 - 2*kappa/dy**2
                    matrix_a[k, k+1] = kappa/dx**2 - rho*cp*u/dx

                if source_mms is not None:
                    rhs[k] = -(f + source_mms(x[j], y[i]))
                else:
                    rhs[k] = -f

    matrix_a = matrix_a.tocsr()
    t_array = spsolve(matrix_a, rhs)

    return t_array


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
# pylint: disable=too-many-branches, too-many-statements, too-many-nested-blocks
def solver_second_order(
        input_dict, scheme='central', sym_test=False, source_mms=None,
        bc_left=None, bc_right=None, bc_bottom=None, bc_top_tinf=None
):
    """
    Résout l'équation de convection-diffusion (schéma ordre 2).

    :param input_dict: paramètres du problème ('nx', 'ny', 'k', 'b', etc.)
    :param scheme: Schéma d'advection ('central' ou 'upwind')
    :param sym_test: booléen (True si test de symétrie)
    :param source_mms: fonction source supplémentaire pour le test MMS
    :param bc_left, bc_right, bc_bottom, bc_top_tinf: conditions aux limites
    :return: tableau 1D de la température
    """
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

    n_nodes = nx * ny
    matrix_a = lil_matrix((n_nodes, n_nodes))
    rhs = np.zeros(nx*ny)

    if scheme == 'central':
        for i in range(ny):
            for j in range(nx):
                k = i * nx + j

                if j == 0:
                    matrix_a[k, k] = 1
                    if bc_left is None:
                        rhs[k] = temp_a
                    else:
                        rhs[k] = bc_left(y[i])

                elif j == nx-1:
                    matrix_a[k, k] = 1
                    if bc_right is None:
                        rhs[k] = temp_b
                    else:
                        rhs[k] = bc_right(y[i])

                elif not sym_test and i == 0:
                    if bc_bottom is None:
                        matrix_a[k, k] = -3
                        matrix_a[k, k+nx] = 4
                        matrix_a[k, k+2*nx] = -1
                        rhs[k] = 0
                    else:
                        matrix_a[k, k] = -3.0 / (2*dy)
                        matrix_a[k, k + nx] = 4.0 / (2*dy)
                        matrix_a[k, k + 2*nx] = -1.0 / (2*dy)
                        rhs[k] = bc_bottom(x[j])

                elif sym_test and i == 0:
                    matrix_a[k, k] = -3*kappa - 2*dy*h
                    matrix_a[k, k+nx] = 4*kappa
                    matrix_a[k, k+2*nx] = -1*kappa
                    rhs[k] = -2*dy*h*tinf

                elif i == ny-1:
                    matrix_a[k, k] = 3*kappa + 2*dy*h
                    matrix_a[k, k-nx] = -4*kappa
                    matrix_a[k, k-2*nx] = 1*kappa
                    if bc_top_tinf is None:
                        rhs[k] = 2*dy*h*tinf
                    else:
                        rhs[k] = 2*dy*h*bc_top_tinf(x[j])

                else:
                    u = speed_function(c, d, y[i])
                    matrix_a[k, k-1] = rho*cp*u/(2*dx) + kappa/dx**2
                    matrix_a[k, k] = -2*kappa/dx**2 - 2*kappa/dy**2
                    matrix_a[k, k+1] = kappa/dx**2 - rho*cp*u/(2*dx)
                    matrix_a[k, k-nx] = kappa/dy**2
                    matrix_a[k, k+nx] = kappa/dy**2
                    if source_mms is None:
                        rhs[k] = -f
                    else:
                        rhs[k] = -(f + source_mms(x[j], y[i]))

    elif scheme == 'upwind':
        for i in range(ny):
            for j in range(nx):
                k = i * nx + j

                if j == 0:
                    matrix_a[k, k] = 1
                    if bc_left is None:
                        rhs[k] = temp_a
                    else:
                        rhs[k] = bc_left(y[i])

                elif j == nx-1:
                    matrix_a[k, k] = 1
                    if bc_right is None:
                        rhs[k] = temp_b
                    else:
                        rhs[k] = bc_right(y[i])

                elif not sym_test and i == 0:
                    if bc_bottom is None:
                        matrix_a[k, k] = -3
                        matrix_a[k, k+nx] = 4
                        matrix_a[k, k+2*nx] = -1
                        rhs[k] = 0
                    else:
                        matrix_a[k, k] = -3.0 / (2*dy)
                        matrix_a[k, k + nx] = 4.0 / (2*dy)
                        matrix_a[k, k + 2*nx] = -1.0 / (2*dy)
                        rhs[k] = bc_bottom(x[j])

                elif sym_test and i == 0:
                    matrix_a[k, k] = -3*kappa - 2*dy*h
                    matrix_a[k, k+nx] = 4*kappa
                    matrix_a[k, k+2*nx] = -1*kappa
                    rhs[k] = -2*dy*h*tinf

                elif i == ny-1:
                    matrix_a[k, k] = 3*kappa + 2*dy*h
                    matrix_a[k, k-nx] = -4*kappa
                    matrix_a[k, k-2*nx] = 1*kappa
                    if bc_top_tinf is None:
                        rhs[k] = 2*dy*h*tinf
                    else:
                        rhs[k] = 2*dy*h*bc_top_tinf(x[j])

                else:
                    u = speed_function(c, d, y[i])
                    matrix_a[k, k-nx] = kappa/dy**2
                    matrix_a[k, k+nx] = kappa/dy**2

                    if u >= 0:
                        if j == 1:
                            matrix_a[k, k-1] = kappa/dx**2 + rho*cp*u/(2*dx)
                            matrix_a[k, k] = -2*kappa/dx**2 - 2*kappa/dy**2
                            matrix_a[k, k+1] = kappa/dx**2 - rho*cp*u/(2*dx)
                        else:
                            matrix_a[k, k-2] = -rho*cp*u/(2*dx)
                            matrix_a[k, k-1] = kappa/dx**2 + 4*rho*cp*u/(2*dx)
                            matrix_a[k, k] = -2*kappa/dx**2 - 2*kappa/dy**2 - 3*rho*cp*u/(2*dx)
                            matrix_a[k, k+1] = kappa/dx**2
                    else:
                        if j == nx-2:
                            matrix_a[k, k-1] = kappa/dx**2 + rho*cp*u/(2*dx)
                            matrix_a[k, k] = -2*kappa/dx**2 - 2*kappa/dy**2
                            matrix_a[k, k+1] = kappa/dx**2 - rho*cp*u/(2*dx)
                        else:
                            matrix_a[k, k-1] = kappa/dx**2
                            matrix_a[k, k] = -2*kappa/dx**2 - 2*kappa/dy**2 + 3*rho*cp*u/(2*dx)
                            matrix_a[k, k+1] = kappa/dx**2 - 4*rho*cp*u/(2*dx)
                            matrix_a[k, k+2] = rho*cp*u/(2*dx)

                    if source_mms is None:
                        rhs[k] = -f
                    else:
                        rhs[k] = -(f + source_mms(x[j], y[i]))

    matrix_a = matrix_a.tocsr()
    t_array = spsolve(matrix_a, rhs)

    return t_array


def mms_temperature(input_dict, mms_func):
    """
    Génère le champ de température exact de la MMS sur le maillage.
    """
    ny = input_dict['ny']
    nx = input_dict['nx']
    b = input_dict['b']
    c = input_dict['c']

    x = np.linspace(0, b, nx)
    y = np.linspace(0, c, ny)

    t_mms_vec = np.zeros(nx*ny)

    for i in range(ny):
        for j in range(nx):
            k = i*nx + j
            t_mms_vec[k] = mms_func(x[j], y[i])

    return t_mms_vec


def save_as_csv(t_array, input_dict, filename='results/temperature_field.csv'):
    """
    Sauvegarde le champ de température dans un fichier CSV.
    """
    b = input_dict['b']
    c = input_dict['c']
    nx = input_dict['nx']
    ny = input_dict['ny']

    x = np.linspace(0, b, nx)
    y = np.linspace(0, c, ny)

    t_mesh = t_array.reshape((ny, nx))

    data = np.zeros((ny*nx, 3))
    idx = 0
    for i in range(ny):
        for j in range(nx):
            data[idx] = [x[j], y[i], t_mesh[i, j]]
            idx += 1

    np.savetxt(filename, data, delimiter=',', header='x,y,T', comments='')


def save_input_as_csv(input_dict, filename='results/input_parameters.csv'):
    """
    Sauvegarde le dictionnaire des paramètres dans un fichier CSV.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for key, value in input_dict.items():
            f.write(f"{key},{value}\n")
