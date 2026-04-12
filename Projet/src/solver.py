"""
Module des solveurs numériques.

Contient les solveurs par différences finies d'ordre 1 et 2,
ainsi que les fonctions de calcul de flux et de conservation.
"""
import numpy as np
from scipy.sparse import lil_matrix, diags
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
    Résout l'équation de convection-diffusion via une approche vectorisée hybride.
    """
    b, c, d = input_dict['b'], input_dict['c'], input_dict['d']
    nx, ny = input_dict['nx'], input_dict['ny']
    rho, cp, kappa, f = input_dict['rho'], input_dict['cp'], input_dict['k'], input_dict['f']
    temp_a, temp_b = input_dict['temp_a'], input_dict['temp_b']
    h, tinf = input_dict['h'], input_dict['tinf']

    x = np.linspace(0, b, nx)
    y = np.linspace(-c, c, ny) if sym_test else np.linspace(0, c, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    n_nodes = nx * ny

    # --- 1. PRÉ-CALCULS VECTORISÉS ---
    x_mesh, y_mesh = np.meshgrid(x, y)
    y_flat = y_mesh.flatten()

    u_flat = speed_function(c, d, y_flat)
    diff_x, diff_y = kappa / dx**2, kappa / dy**2
    adv_flat = rho * cp * u_flat / (2 * dx)

    # Indice j (colonne) pour chaque nœud (0 à nx-1 répété)
    j_flat = np.tile(np.arange(nx), ny)

    # --- 2. CONSTRUCTION DES DIAGONALES (Nœuds intérieurs) ---
    south_diag = np.full(n_nodes - nx, diff_y)
    north_diag = np.full(n_nodes - nx, diff_y)

    if scheme == 'central':
        main_diag = np.full(n_nodes, -2*diff_x - 2*diff_y)
        west_diag = diff_x + adv_flat[1:]
        east_diag = diff_x - adv_flat[:-1]

        diagonals = [main_diag, south_diag, north_diag, west_diag, east_diag]
        offsets = [0, -nx, nx, -1, 1]

    elif scheme == 'upwind':
        # Puisque u >= 0 partout, on applique Upwind à droite.
        # j_flat == 1 -> Centré. j_flat > 1 -> Upwind.
        main_diag = np.where(j_flat > 1, -2*diff_x - 2*diff_y - 3*adv_flat, -2*diff_x - 2*diff_y)
        west_diag = np.where(j_flat[1:] > 1, diff_x + 4*adv_flat[1:], diff_x + adv_flat[1:])
        east_diag = np.where(j_flat[:-1] == 1, diff_x - adv_flat[:-1], diff_x)
        ww_diag = np.where(j_flat[2:] > 1, -adv_flat[2:], 0.0)

        diagonals = [main_diag, south_diag, north_diag, west_diag, east_diag, ww_diag]
        offsets = [0, -nx, nx, -1, 1, -2]

    # Génération instantanée de la matrice creuse avec diags
    matrix_a = diags(diagonals, offsets, shape=(n_nodes, n_nodes), format='lil')

    # --- 3. VECTEUR SOURCE (RHS) ---
    if source_mms is None:
        rhs = np.full(n_nodes, -f, dtype=float)
    else:
        rhs = -(f + source_mms(x_mesh.flatten(), y_flat))

    # --- 4. ÉCRASEMENT DES CONDITIONS AUX LIMITES ---
    # Cette méthode accède directement à l'architecture de lil_matrix pour des perfs maximales

    # A. Gauche (Dirichlet)
    for k in range(0, n_nodes, nx):
        matrix_a.data[k] = [1.0]
        matrix_a.rows[k] = [k]
        rhs[k] = temp_a if bc_left is None else bc_left(y[k // nx])

    # B. Droite (Dirichlet)
    for k in range(nx - 1, n_nodes, nx):
        matrix_a.data[k] = [1.0]
        matrix_a.rows[k] = [k]
        rhs[k] = temp_b if bc_right is None else bc_right(y[k // nx])

    # C. Bas (Neumann/Robin) - excluant les coins (0 et nx-1)
    for k in range(1, nx - 1):
        if sym_test:
            matrix_a.data[k] = [-3*kappa - 2*dy*h, 4*kappa, -kappa]
            matrix_a.rows[k] = [k, k+nx, k+2*nx]
            rhs[k] = -2*dy*h*tinf
        else:
            if bc_bottom is None:
                matrix_a.data[k] = [-3.0, 4.0, -1.0]
                matrix_a.rows[k] = [k, k+nx, k+2*nx]
                rhs[k] = 0.0
            else:
                matrix_a.data[k] = [-3.0/(2*dy), 4.0/(2*dy), -1.0/(2*dy)]
                matrix_a.rows[k] = [k, k+nx, k+2*nx]
                rhs[k] = bc_bottom(x[k])

    # D. Haut (Robin) - excluant les coins
    for k in range(n_nodes - nx + 1, n_nodes - 1):
        matrix_a.data[k] = [3*kappa + 2*dy*h, -4*kappa, kappa]
        matrix_a.rows[k] = [k, k-nx, k-2*nx]
        rhs[k] = 2*dy*h*tinf if bc_top_tinf is None else 2*dy*h*bc_top_tinf(x[k % nx])

    # --- 5. RÉSOLUTION ---
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
