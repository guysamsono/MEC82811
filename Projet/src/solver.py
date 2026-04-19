"""
Module des solveurs numériques.

Contient les solveurs par différences finies d'ordre 1 et 2,
ainsi que les fonctions de calcul de flux et de conservation.
"""
import numpy as np
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import spsolve

def bc_top_tinf_fabriquee(x, input_dict):
    tinf = input_dict['tinf']
    return tinf


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
def compute_conservation_of_energy(t_array, input_dict, source_mms=None):
    """
    Calcule le résidu de conservation d'énergie globale.

    Convention :
    flux sortant positif.

    Bilan :
        ∫_∂Ω (-k grad(T)·n + rho cp T u·n) dΓ
        - ∫_Ω (f + source_mms) dΩ

    Pour la MMS, passer source_mms=f_source.
    """

    ny, nx = input_dict['ny'], input_dict['nx']
    kappa = input_dict['k']
    rho = input_dict['rho']
    cp = input_dict['cp']
    b = input_dict['b']
    c = input_dict['c']
    d = input_dict['d']
    f = input_dict['f']

    t_mesh = np.asarray(t_array).reshape((ny, nx))

    x = np.linspace(0.0, b, nx)
    y = np.linspace(0.0, c, ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    u_y = speed_function(c, d, y)

    # -----------------------------
    # Flux diffusifs aux frontières
    # -----------------------------

    # Gauche x = 0, normale n = (-1, 0)
    # flux diffusif sortant = -k grad(T)·n = k dT/dx
    dTdx_left = (-3.0 * t_mesh[:, 0] + 4.0 * t_mesh[:, 1] - t_mesh[:, 2]) / (2.0 * dx)
    qdiff_left = kappa * dTdx_left

    # Droite x = b, normale n = (1, 0)
    # flux diffusif sortant = -k dT/dx
    dTdx_right = (3.0 * t_mesh[:, -1] - 4.0 * t_mesh[:, -2] + t_mesh[:, -3]) / (2.0 * dx)
    qdiff_right = -kappa * dTdx_right

    # Bas y = 0, normale n = (0, -1)
    # flux diffusif sortant = k dT/dy
    dTdy_bottom = (-3.0 * t_mesh[0, :] + 4.0 * t_mesh[1, :] - t_mesh[2, :]) / (2.0 * dy)
    qdiff_bottom = kappa * dTdy_bottom

    # Haut y = c, normale n = (0, 1)
    # flux diffusif sortant = -k dT/dy
    dTdy_top = (3.0 * t_mesh[-1, :] - 4.0 * t_mesh[-2, :] + t_mesh[-3, :]) / (2.0 * dy)
    qdiff_top = -kappa * dTdy_top

    # -----------------------------
    # Flux convectifs
    # -----------------------------
    # Vitesse u = (u(y), 0)
    # Gauche : u·n = -u(y)
    # Droite : u·n = +u(y)

    qconv_left = rho * cp * t_mesh[:, 0] * (-u_y)
    qconv_right = rho * cp * t_mesh[:, -1] * u_y

    # -----------------------------
    # Intégrales de frontière
    # -----------------------------

    flux_left = np.trapz(qdiff_left + qconv_left, y)
    flux_right = np.trapz(qdiff_right + qconv_right, y)
    flux_bottom = np.trapz(qdiff_bottom, x)
    flux_top = np.trapz(qdiff_top, x)

    boundary_flux = flux_left + flux_right + flux_bottom + flux_top

    # -----------------------------
    # Intégrale volumique de source
    # -----------------------------

    x_mesh, y_mesh = np.meshgrid(x, y)

    if source_mms is None:
        source_total = f * np.ones_like(x_mesh)
    else:
        source_total = f + source_mms(x_mesh, y_mesh)

    # Intégrale 2D trapézoïdale
    source_integral_y = np.trapz(source_total, y, axis=0)
    source_integral = np.trapz(source_integral_y, x)

    residual = boundary_flux - source_integral

    return residual

def compute_average_temperature(t_array, input_dict):
    ny, nx = input_dict['ny'], input_dict['nx']
    b = input_dict['b']
    t_mesh = np.asarray(t_array).reshape((ny, nx))
    dx = b / (nx - 1)

    # Intégration par la méthode des trapèzes (Ordre 2) au lieu de np.mean (Ordre 1)
    average_temp = np.trapezoid(t_mesh[-1, :], dx=dx) / b
    return average_temp


def compute_temperature_at_y(t_array, input_dict, y_ratio=0.8, x_ratio=0.8):
    """
    Calcule la température en un point du domaine situé à
    y = y_ratio * c et x = x_ratio * b
    par interpolation bilinéaire.
    """
    ny = input_dict['ny']
    nx = input_dict['nx']
    b = input_dict['b']
    c = input_dict['c']

    t_mesh = np.asarray(t_array).reshape((ny, nx))

    # Coordonnées physiques du point
    x_target = x_ratio * b
    y_target = y_ratio * c

    dx = b / (nx - 1)
    dy = c / (ny - 1)

    # Cas limites pour éviter les débordements
    x_target = min(max(x_target, 0.0), b)
    y_target = min(max(y_target, 0.0), c)

    # Indices de la cellule contenant le point
    j0 = int(np.floor(x_target / dx))
    i0 = int(np.floor(y_target / dy))

    # Si on tombe sur le dernier nœud, on force la cellule précédente
    if j0 >= nx - 1:
        j0 = nx - 2
    if i0 >= ny - 1:
        i0 = ny - 2

    j1 = j0 + 1
    i1 = i0 + 1

    # Coordonnées des 4 nœuds entourant le point
    x0 = j0 * dx
    x1 = j1 * dx
    y0 = i0 * dy
    y1 = i1 * dy

    # Valeurs aux 4 coins
    t00 = t_mesh[i0, j0]
    t10 = t_mesh[i0, j1]
    t01 = t_mesh[i1, j0]
    t11 = t_mesh[i1, j1]

    # Poids d'interpolation
    if x1 == x0:
        wx = 0.0
    else:
        wx = (x_target - x0) / (x1 - x0)

    if y1 == y0:
        wy = 0.0
    else:
        wy = (y_target - y0) / (y1 - y0)

    # Interpolation bilinéaire
    temperature = (
        (1 - wx) * (1 - wy) * t00 +
        wx * (1 - wy) * t10 +
        (1 - wx) * wy * t01 +
        wx * wy * t11
    )

    return temperature


def compute_boundary_fluxes(
        t_array,
        input_dict,
        x_start_ratio=0.0,
        x_end_ratio=1.0
):
    """
    Calcule le flux de chaleur à travers une portion de la frontière supérieure.

    La portion intégrée est définie par :

        x_start = x_start_ratio * b
        x_end   = x_end_ratio * b

    Exemples :
        x_start_ratio=0.2, x_end_ratio=0.8  -> portion centrale 60 %
        x_start_ratio=0.0, x_end_ratio=1.0  -> frontière complète
        x_start_ratio=0.1, x_end_ratio=0.4  -> portion non centrée à gauche
        x_start_ratio=0.6, x_end_ratio=0.9  -> portion non centrée à droite

    Les bornes sont interpolées si elles ne tombent pas exactement sur des nœuds.
    """

    ny, nx = input_dict['ny'], input_dict['nx']
    kappa = input_dict['k']
    b = input_dict['b']
    c = input_dict['c']

    t_mesh = np.asarray(t_array).reshape((ny, nx))

    dy = c / (ny - 1)

    x_nodes = np.linspace(0.0, b, nx)

    # Flux diffusif sur tous les nœuds du bord supérieur
    flux_top = -kappa * (
        3.0 * t_mesh[-1, :]
        - 4.0 * t_mesh[-2, :]
        + t_mesh[-3, :]
    ) / (2.0 * dy)

    # Sécurité sur les ratios
    if not (0.0 <= x_start_ratio <= 1.0):
        raise ValueError("x_start_ratio doit être entre 0 et 1.")

    if not (0.0 <= x_end_ratio <= 1.0):
        raise ValueError("x_end_ratio doit être entre 0 et 1.")

    if x_end_ratio < x_start_ratio:
        raise ValueError("x_end_ratio doit être supérieur ou égal à x_start_ratio.")

    # Bornes physiques
    x_start = b * x_start_ratio
    x_end = b * x_end_ratio

    # Interpolation linéaire du flux aux bornes
    flux_start = np.interp(x_start, x_nodes, flux_top)
    flux_end = np.interp(x_end, x_nodes, flux_top)

    # Nœuds strictement à l'intérieur de l'intervalle
    mask = (x_nodes > x_start) & (x_nodes < x_end)
    x_interior = x_nodes[mask]
    flux_interior = flux_top[mask]

    # Reconstruction du profil sur l'intervalle exact demandé
    x_eval = np.concatenate(([x_start], x_interior, [x_end]))
    flux_eval = np.concatenate(([flux_start], flux_interior, [flux_end]))

    # Intégration trapézoïdale
    heat_transfer = np.trapezoid(flux_eval, x=x_eval)

    return heat_transfer


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

            # ------------------------------------------------------------
            # 1. Haut : Robin, coins inclus
            # ------------------------------------------------------------
          
            if i == ny - 1:
                matrix_a[k, k] = kappa + h * dy
                matrix_a[k, k - nx] = -kappa

                if bc_top_tinf is None:
                    rhs[k] = h * dy * tinf
                else:
                    rhs[k] = h * dy * bc_top_tinf(x[j])

            # ------------------------------------------------------------
            # 2. Gauche : Dirichlet, sauf coin supérieur gauche
            # ------------------------------------------------------------
            elif j == 0:
                matrix_a[k, k] = 1.0

                if bc_left is None:
                    rhs[k] = temp_a
                else:
                    rhs[k] = bc_left(y[i])

            # ------------------------------------------------------------
            # 3. Droite : Neumann, sauf coin supérieur droit
            # ------------------------------------------------------------
           
            elif j == nx - 1:
                matrix_a[k, k] = 1.0 / dx
                matrix_a[k, k - 1] = -1.0 / dx

                rhs[k] = 0.0 if bc_right is None else bc_right(y[i])

            # ------------------------------------------------------------
            # 4. Bas
            # ------------------------------------------------------------
            elif i == 0:
                if sym_test:
                    matrix_a[k, k] = -(h * dy + kappa)
                    matrix_a[k, k + nx] = kappa
                    rhs[k] = -dy * h * tinf
                else:
                    matrix_a[k, k] = -1.0 / dy
                    matrix_a[k, k + nx] = 1.0 / dy
                    rhs[k] = 0.0 if bc_bottom is None else bc_bottom(x[j])

            # ------------------------------------------------------------
            # 5. Noeuds intérieurs
            # ------------------------------------------------------------
            else:
                u = speed_function(c, d, y[i])

                matrix_a[k, k - nx] = kappa / dy**2
                matrix_a[k, k + nx] = kappa / dy**2

                if u >= 0:
                    matrix_a[k, k - 1] = rho * cp * u / dx + kappa / dx**2
                    matrix_a[k, k] = (
                        -rho * cp * u / dx
                        - 2.0 * kappa / dx**2
                        - 2.0 * kappa / dy**2
                    )
                    matrix_a[k, k + 1] = kappa / dx**2
                else:
                    matrix_a[k, k - 1] = kappa / dx**2
                    matrix_a[k, k] = (
                        rho * cp * u / dx
                        - 2.0 * kappa / dx**2
                        - 2.0 * kappa / dy**2
                    )
                    matrix_a[k, k + 1] = kappa / dx**2 - rho * cp * u / dx

                if source_mms is None:
                    rhs[k] = -f
                else:
                    rhs[k] = -(f + source_mms(x[j], y[i]))

    matrix_a = matrix_a.tocsr()
    t_array = spsolve(matrix_a, rhs)

    return t_array


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
# pylint: disable=too-many-branches, too-many-statements, too-many-nested-blocks
def solver_second_order(
        input_dict, scheme='central', sym_test=False, source_mms=None,
        bc_left=None, bc_bottom=None, bc_top_tinf=None,bc_right=None
):
    """
    Résout l'équation de convection-diffusion via une approche vectorisée hybride.

    Conditions:
    - gauche  : Dirichlet
    - droite  : Neumann nulle
    - bas     : Neumann (ou Robin si sym_test=True)
    - haut    : Robin, coins inclus

    Si bc_top_tinf is None, une Robin fabriquée lisse est utilisée:
        T_inf(x) = Ta + (Tb - Ta) * (3*xi^2 - 2*xi^3), xi = x/b
    """
    b, c, d = input_dict['b'], input_dict['c'], input_dict['d']
    nx, ny = input_dict['nx'], input_dict['ny']
    rho, cp, kappa, f = input_dict['rho'], input_dict['cp'], input_dict['k'], input_dict['f']
    temp_a = input_dict['temp_a']
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

    j_flat = np.tile(np.arange(nx), ny)

    # --- 2. CONSTRUCTION DES DIAGONALES ---
    south_diag = np.full(n_nodes - nx, diff_y)
    north_diag = np.full(n_nodes - nx, diff_y)

    if scheme == 'central':
        main_diag = np.full(n_nodes, -2 * diff_x - 2 * diff_y)
        west_diag = diff_x + adv_flat[1:]
        east_diag = diff_x - adv_flat[:-1]

        diagonals = [main_diag, south_diag, north_diag, west_diag, east_diag]
        offsets = [0, -nx, nx, -1, 1]

    elif scheme == 'upwind':
        main_diag = np.where(
            j_flat > 1,
            -2 * diff_x - 2 * diff_y - 3 * adv_flat,
            np.where(
                j_flat == 1,
                -2 * diff_x - 2 * diff_y - 2 * adv_flat,
                -2 * diff_x - 2 * diff_y
            )
        )

        west_diag = np.where(
            j_flat[1:] > 1,
            diff_x + 4 * adv_flat[1:],
            diff_x + 2 * adv_flat[1:]
        )

        east_diag = np.full(n_nodes - 1, diff_x)
        ww_diag = np.where(j_flat[2:] > 1, -adv_flat[2:], 0.0)

        diagonals = [main_diag, south_diag, north_diag, west_diag, east_diag, ww_diag]
        offsets = [0, -nx, nx, -1, 1, -2]

    else:
        raise ValueError("scheme doit être 'central' ou 'upwind'")

    matrix_a = diags(diagonals, offsets, shape=(n_nodes, n_nodes), format='lil')

    # --- 3. RHS ---
    if source_mms is None:
        rhs = np.full(n_nodes, -f, dtype=float)
    else:
        rhs = -(f + source_mms(x_mesh.flatten(), y_flat))

    # --- 4. CONDITIONS AUX LIMITES ---

    # A. Gauche (Dirichlet), sauf coin supérieur gauche
    for i in range(ny - 1):
        k = i * nx
        matrix_a.data[k] = [1.0]
        matrix_a.rows[k] = [k]
        rhs[k] = temp_a if bc_left is None else bc_left(y[i])

    # B. Droite (Neumann nulle), sauf coin supérieur droit
    for i in range(ny - 1):
        k = i * nx + (nx - 1)
        matrix_a.data[k] = [3.0 / (2 * dx), -4.0 / (2 * dx), 1.0 / (2 * dx)]
        matrix_a.rows[k] = [k, k - 1, k - 2]
        rhs[k] = 0.0 if bc_right is None else bc_right(y[i])

    # C. Bas, excluant les coins du bas
    for j in range(1, nx - 1):
        k = j
        if sym_test:
            matrix_a.data[k] = [-3 * kappa - 2 * dy * h, 4 * kappa, -kappa]
            matrix_a.rows[k] = [k, k + nx, k + 2 * nx]
            rhs[k] = -2 * dy * h * tinf
        else:
            if bc_bottom is None:
                matrix_a.data[k] = [-3.0, 4.0, -1.0]
                matrix_a.rows[k] = [k, k + nx, k + 2 * nx]
                rhs[k] = 0.0
            else:
                matrix_a.data[k] = [-3.0 / (2 * dy), 4.0 / (2 * dy), -1.0 / (2 * dy)]
                matrix_a.rows[k] = [k, k + nx, k + 2 * nx]
                rhs[k] = bc_bottom(x[j])

    # D. Haut (Robin fabriquée), partout, coins inclus
    for j in range(nx):
        k = (ny - 1) * nx + j
        matrix_a.data[k] = [3 * kappa + 2 * dy * h, -4 * kappa, kappa]
        matrix_a.rows[k] = [k, k - nx, k - 2 * nx]

        if bc_top_tinf is None:
            rhs[k] = 2 * dy * h * bc_top_tinf_fabriquee(x[j], input_dict)
        else:
            rhs[k] = 2 * dy * h * bc_top_tinf(x[j])

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

    x_mesh, y_mesh = np.meshgrid(x, y)
    t_mms_matrix = mms_func(x_mesh, y_mesh)
    t_mms_vec = t_mms_matrix.flatten()

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
