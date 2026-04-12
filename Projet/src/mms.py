"""
Module de Vérification de Code par la Méthode des Solutions Manufacturées (MMS).

Ce module permet de vérifier la validité de l'implémentation numérique en comparant
la solution calculée par le solveur à une solution analytique arbitraire (manufacturée).
Il gère la génération symbolique du terme source, des conditions aux limites
correspondantes et l'analyse de l'ordre de convergence formel.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from src.solver import solver_first_order, solver_second_order, mms_temperature
from src.error import norm_l1, norm_l2, norm_infinity
from src.convergence import graph_error_log, print_convergence_table


def generer_mms_simple(input_dict: dict):
    """
    Génère la solution manufacturée, le terme source et les graphiques associés.

    :param input_dict: Dictionnaire des paramètres physiques et d'entrée du problème.
    :return: f_t_mms, f_source, f_bc_left, f_bc_right, f_bc_bottom, f_tinf_top
    """
    # pylint: disable=too-many-locals, too-many-statements

    # Extraction des paramètres physiques
    ny = input_dict['ny']
    nx = input_dict['nx']
    kappa = input_dict['k']
    rho = input_dict['rho']
    cp = input_dict['cp']
    b = input_dict['b']
    c = input_dict['c']
    d = input_dict['d']
    f = input_dict['f']
    h = input_dict['h']

    # Définition des variables symboliques
    x, y = sp.symbols('x y')

    # Solution manufacturée (parenthèses ajoutées pour respecter la limite de ligne)
    t_mms = (100 + sp.sin(sp.pi * x / b) + sp.cos(sp.pi * y / (2 * c)) +
             sp.sin(2 * sp.pi * x / b) * sp.cos(2 * sp.pi * y / (2 * c)))

    # Calcul des dérivées
    t_x = sp.diff(t_mms, x)
    t_y = sp.diff(t_mms, y)
    t_xx = sp.diff(t_x, x)
    t_yy = sp.diff(t_y, y)

    # Calcul du terme source S(x,y)
    u_sym = (3 * d) / (4 * c) * (1 - (y / c)**2)
    source = rho * cp * u_sym * t_x - kappa * (t_xx + t_yy) - f

    # Conditions frontières
    # Gamma 2 : x = 0
    t_boundary_2 = sp.simplify(t_mms.subs(x, 0))

    # Gamma 4 : x = b
    t_boundary_4 = sp.simplify(t_mms.subs(x, b))

    # Gamma 3 : y = 0, Neumann = dT/dy(x,0)
    dt_dy_boundary_3 = sp.simplify(sp.diff(t_mms, y).subs(y, 0))

    # Gamma 1 : y = c, Robin
    t_top = sp.simplify(t_mms.subs(y, c))
    dt_dy_top = sp.simplify(sp.diff(t_mms, y).subs(y, c))
    t_inf_top = sp.simplify(t_top + (kappa / h) * dt_dy_top)

    # Conversion en fonctions Python
    f_t_mms = sp.lambdify((x, y), t_mms, "numpy")
    f_source = sp.lambdify((x, y), source, "numpy")
    f_bc_left = sp.lambdify(y, t_boundary_2, "numpy")
    f_bc_right = sp.lambdify(y, t_boundary_4, "numpy")
    f_bc_bottom = sp.lambdify(x, dt_dy_boundary_3, "numpy")
    f_tinf_top = sp.lambdify(x, t_inf_top, "numpy")

    # Création des maillages spatial pour les graphiques
    xdom = np.linspace(0, b, nx)
    ydom = np.linspace(0, c, ny)
    xi, yi = np.meshgrid(xdom, ydom, indexing='ij')

    # Évaluation des fonctions sur le maillage pour les graphiques
    z_mms = f_t_mms(xi, yi)
    z_source = f_source(xi, yi)

    # Graphiques
    mms_dir = os.path.join(input_dict['save_path'], "MMS")
    os.makedirs(mms_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    contour1 = plt.contourf(xdom, ydom, z_mms.T, 100, cmap='hot')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(contour1, label='Temperature')
    plt.title('Temperature distribution for MMS solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(os.path.join(mms_dir, "MMS_solution.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    contour2 = plt.contourf(xdom, ydom, z_source.T, 100, cmap='viridis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(contour2, label='Terme source S(x,y) [W/m³]')
    plt.title('Source distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(os.path.join(mms_dir, "MMS_source.png"), dpi=300)
    plt.close()

    return f_t_mms, f_source, f_bc_left, f_bc_right, f_bc_bottom, f_tinf_top


def mms_convergence_analysis(input_dict: dict, order, scheme='central'):
    """
    Réalise une analyse de convergence spatiale via la MMS.

    :param input_dict: Dictionnaire des paramètres physiques et d'entrée.
    :param order: Ordre formel du schéma de discrétisation à tester ('1' ou '2').
    :param scheme: Schéma de différences finies utilisé pour le terme d'advection.
    """
    # pylint: disable=too-many-locals

    local_dict = input_dict.copy()

    maille_list = [100, 200, 300]
    discretization_list = [local_dict['b'] / (nx - 1) for nx in maille_list]

    l1_list_x = []
    l2_list_x = []
    linf_list_x = []

    for n in maille_list:
        local_dict['nx'] = n
        local_dict['ny'] = n

        # Utilisation de parenthèses ou de backslash pour couper les lignes trop longues
        f_t_mms, f_source, f_bc_left, f_bc_right, f_bc_bottom, f_tinf_top = \
            generer_mms_simple(local_dict)

        if order == '1':
            temperature_sim = solver_first_order(
                local_dict, sym_test=False, source_mms=f_source,
                bc_left=f_bc_left, bc_right=f_bc_right,
                bc_bottom=f_bc_bottom, bc_top_tinf=f_tinf_top
            )
        else:
            temperature_sim = solver_second_order(
                local_dict, scheme, sym_test=False, source_mms=f_source,
                bc_left=f_bc_left, bc_right=f_bc_right,
                bc_bottom=f_bc_bottom, bc_top_tinf=f_tinf_top
            )

        temperature_mms = mms_temperature(local_dict, f_t_mms)

        l1_list_x.append(norm_l1(temperature_sim, temperature_mms))
        l2_list_x.append(norm_l2(temperature_sim, temperature_mms))
        linf_list_x.append(norm_infinity(temperature_sim, temperature_mms))

    graph_error_log(
        local_dict, discretization_list, l1_list_x, l2_list_x, linf_list_x, 1,
        'x', file_name=f"convergence_x_order_{order}.png",
        show_fig=False, xlabel=r"Taille de maille"
    )

    print_convergence_table(maille_list, discretization_list, l1_list_x, str(order), "L1 en x")
    print_convergence_table(maille_list, discretization_list, l2_list_x, str(order), "L2 en x")
    print_convergence_table(maille_list, discretization_list, linf_list_x, str(order), "Linf en x")
