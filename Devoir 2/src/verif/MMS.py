"""
Module de calcul de la MMS (conditions frontières)
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def generer_mms(params: dict, nt: int, nr: int, afficher_graphiques: bool = False, save_path: str = "results/temp"):
    """
    Génère la solution manufacturée, le terme source et les graphiques associés.
    
    :param params: Dictionnaire des paramètres physiques
    :param nt: Nombre de nœuds temporels (discrétisation en temps)
    :param nr: Nombre de nœuds spatiaux (discrétisation en espace)
    :param afficher_graphiques: Booléen pour bloquer/débloquer l'affichage des graphiques (désactiver pour le Bash)
    :return: f_C_MMS, f_source (fonctions lambdifiées utilisables par les solveurs)
    """
    # Extraction des paramètres physiques
    Ce = params.get("CE", 20)
    R = params.get("RO", 0.5)
    D_eff = params.get("D_EFF", 1e-10)
    K = params.get("K", 4e-9)
    
    tmin = 0
    tmax = params.get("TF", 100)
    rmin = params.get("RI", 0)
    rmax = R

    # Définition des variables symboliques
    t, r = sp.symbols('t r')

    # Solution manufacturée
    C_MMS = Ce + 10*sp.exp(-t*0.001)*(1-(r/R)**2)  

    # Calcul des dérivées
    C_t = sp.diff(C_MMS, t)
    C_r = sp.diff(C_MMS, r)
    C_rr = sp.diff(C_MMS, r, r)

    # Calcul du terme source S(t,r)
    source = C_t - D_eff*(C_rr + (1/r)*C_r) + K*C_MMS

    # Conditions aux limites et initiales
    C_initial = C_MMS.subs(t, 0)
    C_boundary_re = C_MMS.subs(r, R)
    dCdr_boundary_ri = sp.diff(C_MMS, r).subs(r, 0)

    # Affichage des dérivées
    print("Dérivée en temps :")
    print(C_t)
    print("Dérivée première :")
    print(C_r)
    print("Dérivée seconde :")
    print(C_rr)
    print("Terme source :")
    print(source)
    print("\nCondition initiale T(0, r) :")
    print(C_initial)
    print("\nCondition frontière T(t, R) :")
    print(C_boundary_re)
    print("\nCondition frontière Neumann dT/dr(t, 0) :")
    print(dCdr_boundary_ri)

    # Conversion en fonctions Python
    f_C_MMS = sp.lambdify([t, r], C_MMS, "numpy")
    f_source = sp.lambdify([t, r], source, "numpy")

    # Création des maillages temporel et spatial
    tdom = np.linspace(tmin, tmax, nt)
    rdom = np.linspace(rmin, rmax, nr)
    ti, ri = np.meshgrid(tdom, rdom, indexing='ij')

    # Évaluation des fonctions sur le maillage pour les graphiques
    z_MMS = f_C_MMS(ti, ri)
    z_source = f_source(ti, ri)  

    # Graphique
    plt.figure()
    plt.contourf(ri, ti, z_MMS, levels=50)
    plt.colorbar()
    plt.title('Solution Manufacturée')
    plt.xlabel('r')
    plt.ylabel('t')
    plt.savefig(f"{save_path}_manufactured.png", dpi=300)

    plt.figure()
    plt.contourf(ri, ti, z_source, levels=50)
    plt.colorbar()
    plt.title('Terme Source')
    plt.xlabel('r')
    plt.ylabel('t')
    plt.savefig(f"{save_path}_source.png", dpi=300)


    return f_C_MMS, f_source