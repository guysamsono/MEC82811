"""
Fonctions fournissant les solver numériques et analytique pour le problème de diffusion.
"""
import numpy as np
from scipy.linalg import solve_banded

def first_order(params:dict, n_points=100, f_source=None, f_exacte=None):
    """
    Schéma de résolution d'ordre 1.

    :param params: paramètres du problème
    :param n_points: nombre de noeuds
    :param f_source: Fonction générant le terme source MMS S(t, r)
    :param f_exacte: Fonction de la solution exacte MMS pour la condition initiale
    """
    ri = params["RI"]
    ro = params["RO"]
    ce = params["CE"]
    d_eff = params["D_EFF"]
    k = params["K"]
    dt = params["DT"]
    tf = params["TF"]

    discretization = np.linspace(ri, ro, n_points)
    dr = discretization[1] - discretization[0]
    temps = 0

    # Condition Initiale
    if f_exacte is not None:
        concentration_vect = f_exacte(0, discretization)
    else:
        concentration_vect = np.zeros(n_points)

    # Historique 2D pour le calcul d'erreur
    historique_concentration = [concentration_vect.copy()]
    vecteur_temps = [temps]

    while temps < tf:
        temps += dt  
        
        lower = np.zeros(n_points - 1)
        diag = np.zeros(n_points)
        upper = np.zeros(n_points - 1)
        b = np.zeros(n_points) 

        diag[0] = -1.0
        upper[0] = 1.0
        b[0] = 0

        diag[-1] = 1.0
        b[-1] = ce

        lam = d_eff*dt/(dr**2)
        for i in range(1, n_points - 1):
            r_i = discretization[i]
            mu = d_eff*dt/(r_i*dr)

            lower[i - 1] = -lam
            diag[i] = 1 + k * dt + 2 * lam + mu
            upper[i] = -lam - mu
            
            # Évaluation et ajout du terme source
            source_val = 0
            if f_source is not None:
                source_val = f_source(temps, r_i)
                
            b[i] = concentration_vect[i] + (source_val * dt)

        ab = np.zeros((3, n_points))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower

        concentration_vect = solve_banded((1, 1), ab, b)
        
        # Sauvegarde de l'état
        historique_concentration.append(concentration_vect.copy())
        vecteur_temps.append(temps)

    return discretization, np.array(vecteur_temps), np.array(historique_concentration)


def second_order(params:dict, n_points=100, f_source=None, f_exacte=None):
    """
    Schéma de résolution d'ordre 2.

    :param params: paramètres du problème
    :param n_points: nombre de noeuds
    :param f_source: Fonction générant le terme source MMS S(t, r)
    :param f_exacte: Fonction de la solution exacte MMS pour la condition initiale
    """
    ri = params["RI"]
    ro = params["RO"]
    ce = params["CE"]
    d_eff = params["D_EFF"]
    K = params["K"]
    dt = params["DT"]
    tf = params["TF"]

    temps = 0
    discretization = np.linspace(ri, ro, n_points)
    dr = discretization[1] - discretization[0]

    # Condition Initiale
    if f_exacte is not None:
        concentration_vect = f_exacte(0, discretization)
    else:
        concentration_vect = np.zeros(n_points)

    # Historique 2D pour le calcul d'erreur
    historique_concentration = [concentration_vect.copy()]
    vecteur_temps = [temps]

    while temps < tf:
        temps += dt # Avancement au temps futur (Schéma implicite)
        
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
            
            # Évaluation et ajout du terme source
            source_val = 0
            if f_source is not None:
                source_val = f_source(temps, r_i)
                
            b[i] = concentration_vect[i] + (source_val * dt)
            
        concentration_vect = np.linalg.solve(a, b)
        
        # Sauvegarde de l'état
        historique_concentration.append(concentration_vect.copy())
        vecteur_temps.append(temps)
        
    return discretization, np.array(vecteur_temps), np.array(historique_concentration)
