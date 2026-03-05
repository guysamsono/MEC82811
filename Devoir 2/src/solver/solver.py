"""
Fonctions fournissant les solver numériques et analytique pour le problème de diffusion.
"""
import numpy as np

def first_order(params:dict, n_points=100):
    """
    Schéma de résolution d'ordre 1.

    :param params: paramètres du problème
    :param n_points: nombre de noeuds
    """
    ri = params["RI"]
    ro = params["RO"]
    s = params["S"]
    ce = params["CE"]
    d_eff = params["D_EFF"]
    discretization = np.linspace(ri, ro, n_points)
    dr = discretization[1] - discretization[0]

    a = np.zeros((n_points, n_points))
    b = np.ones(n_points) * s

    if ri == 0:
        a[0, 0] = -1
        a[0, 1] = 1
        b[0] = 0

    else:
        b[0] = ce
        a[0, 0] = 1

    a[-1, -1] = 1
    b[-1] = ce

    for i in range(1, n_points - 1):
        r_i = discretization[i]
        a[i, i-1] = d_eff / (dr**2)
        a[i, i] = -2 * d_eff / (dr**2) - d_eff / (r_i * dr)
        a[i, i+1] = d_eff / (r_i * dr) + d_eff / (dr**2)

    concentration_vect = np.linalg.solve(a, b)

    return discretization, concentration_vect


def second_order(params:dict, n_points=100):
    """
    Schéma de résolution d'ordre 2.

    :param params: paramètres du problème
    :param n_points: nombre de noeuds
    """
    ri = params["RI"]
    ro = params["RO"]
    s = params["S"]
    ce = params["CE"]
    d_eff = params["D_EFF"]
    discretization = np.linspace(ri, ro, n_points)
    dr = discretization[1] - discretization[0]

    a = np.zeros((n_points, n_points))
    b = np.ones(n_points) * s

    if ri == 0:
        a[0, 0] = -3
        a[0, 1] = 4
        a[0, 2] = -1
        b[0] = 0.0

    else:
        b[0] = ce
        a[0, 0] = 1

    a[-1, -1] = 1
    b[-1] = ce

    for i in range(1, n_points - 1):
        r_i = discretization[i]
        a[i, i-1] = d_eff / (dr**2) - d_eff / (2 * r_i * dr)
        a[i, i] = -2 * d_eff / (dr**2)
        a[i, i+1] = d_eff / (2 * r_i * dr) + d_eff / (dr**2)

    concentration_vect = np.linalg.solve(a, b)

    return discretization, concentration_vect


def analytique(params:dict, n_points=100):
    """
    Solution analytique du problème de diffusion.

    :param params: paramètres du problème
    :param n_points: nombre de noeuds
    """
    ri = params["RI"]
    ro = params["RO"]
    s = params["S"]
    ce = params["CE"]
    d_eff = params["D_EFF"]
    discretization = np.linspace(ri, ro, n_points)
    concentration_vect = 0.25 * s * ro**2 * (discretization**2 / ro**2 - 1) / d_eff + ce
    return discretization, concentration_vect
