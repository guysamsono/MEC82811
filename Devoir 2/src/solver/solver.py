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
    K = params["K"]
    dt = params["DT"]
    tf = params["TF"]

    discretization = np.linspace(ri, ro, n_points)
    dr = discretization[1] - discretization[0]
    concentration_vect = np.zeros(n_points)
    temps = 0

    while temps < tf:
        a = np.zeros((n_points, n_points))
        b = np.zeros(n_points) 

        a[0, 0] = -1
        a[0, 1] = 1
        b[0] = 0

        a[-1, -1] = 1
        b[-1] = ce

        lam = d_eff*dt/(dr**2)
        for i in range(1, n_points - 1):
            r_i = discretization[i]
            mu = d_eff*dt/(r_i*dr)

            a[i, i-1] = -lam
            a[i, i] = 1 + K*dt + 2*lam + mu
            a[i, i+1] = - lam - mu
            b[i] = concentration_vect[i]

        concentration_vect = np.linalg.solve(a, b)
        temps += dt

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
    K = params["K"]
    dt = params["DT"]
    tf = params["TF"]


    temps = 0
    discretization = np.linspace(ri, ro, n_points)
    dr = discretization[1] - discretization[0]
    concentration_vect = np.zeros(n_points)

    while temps < tf:
        
        a = np.zeros((n_points, n_points))
        b = np.zeros(n_points)

        a[0, 0] = -3
        a[0, 1] = 4
        a[0, 2] = -1
        b[0] = 0.0


        a[-1, -1] = 1
        b[-1] = ce 

        lam = d_eff*dt/(dr**2)

        for i in range(1, n_points - 1):
            r_i = discretization[i]
            mu = d_eff*dt/(2*r_i*dr)

            a[i, i-1] = -lam + mu
            a[i, i] = 1 + K*dt + 2*lam
            a[i, i+1] = -lam - mu
            b[i] = concentration_vect[i]
            
        concentration_vect = np.linalg.solve(a, b)
        temps += dt
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
