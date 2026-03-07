"""
Fichier roulant le code nécessaire pour le devoir (Version Automatisée pour Bash).
"""
import os
import numpy as np
from src.solver.solver import first_order, second_order
from src.verif.error import norm_l2
from src.verif.MMS import generer_mms

params = {
    "RI": 0,
    "RO": 0.5,
    "S": 2e-8,
    "D_EFF": 1e-10,
    "CE": 20,
    "K": 4e-9,
    "TF": 100
}

if __name__ == "__main__":
    # Création robuste du dossier results
    dossier_courant = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(dossier_courant, "results"), exist_ok=True)

    # Valeurs remplacées par sed pour le script Bash
    noeuds_spatiaux = YYYY
    noeuds_temporels = ZZZZ

    # Ajout du paramètre de pas de temps (DT) au dictionnaire
    params["DT"] = params["TF"] / noeuds_temporels

    # Appel de la fonction MMS pour générer les fonctions analytiques
    f_C_exacte, f_terme_source = generer_mms(
        params=params, 
        nr=noeuds_spatiaux, 
        nt=noeuds_temporels, 
        afficher_graphiques=False
    )

    # Appel du solveur numérique
    discretisation, concentration_num = first_order(params, noeuds_spatiaux)

    # Évaluation de la solution analytique exacte
    # Le solveur actuel (solver.py) retourne la concentration à l'instant final TF
    # On évalue donc la solution MMS exacte à t = TF pour comparer
    concentration_exacte = f_C_exacte(params["TF"], discretisation)

    # Calcul de l'erreur
    erreur_l2 = norm_l2(concentration_num, concentration_exacte)

    # Impression de l'erreur pour le script bash (NE PAS MODIFIER)
    # La syntaxe doit correspondre exactement à ce que grep cherche
    print(f"The absolute error = {erreur_l2}")