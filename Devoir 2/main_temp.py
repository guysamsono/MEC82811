"""
Fichier roulant le code nécessaire pour le devoir (Version Automatisée pour Bash).
"""
import os
import numpy as np
from src.solver.solver import first_order, second_order
from src.verif.error import norm_l1, norm_l2, norm_infinity
from src.verif.MMS import generer_mms

params = {
    "RI": 0,
    "RO": 0.5,
    "S": 2e-8,
    "D_EFF": 1e-2,
    "CE": 20,
    "K": 4e-2,
    "TF": 1
}

if __name__ == "__main__":
    # Création robuste du dossier results
    dossier_courant = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(dossier_courant, "results"), exist_ok=True)

    intervales_spatiaux = 100
    intervales_temporels = 100

    # Ajout du paramètre de pas de temps (DT) au dictionnaire
    params["DT"] = params["TF"] / intervales_temporels

    # Appel de la fonction MMS pour générer les fonctions analytiques
    f_C_exacte, f_terme_source = generer_mms(
        params=params, 
        nr=intervales_spatiaux, 
        nt=intervales_temporels, 
        afficher_graphiques=False
    )

    # Appel du solveur numérique
    discretisation, tableau_temps, concentration_num_2d = first_order(
        params, 
        intervales_spatiaux, 
        f_source=f_terme_source, 
        f_exacte=f_C_exacte
    )
    
    # Évaluation de la solution analytique exacte sur TOUT le domaine spatio-temporel
    grille_temps, grille_espace = np.meshgrid(tableau_temps, discretisation, indexing='ij')
    concentration_exacte_2d = f_C_exacte(grille_temps, grille_espace)

    # Calcul des trois normes
    err_l1 = norm_l1(concentration_num_2d, concentration_exacte_2d)
    err_l2 = norm_l2(concentration_num_2d, concentration_exacte_2d)
    err_inf = norm_infinity(concentration_num_2d, concentration_exacte_2d)

    # Impression de l'erreur pour la lecture bash (NE PAS MODIFIER)
    print(f"Error L1 = {err_l1}")
    print(f"Error L2 = {err_l2}")
    print(f"Error Linf = {err_inf}")