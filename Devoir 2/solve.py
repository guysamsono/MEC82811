"""
Fichier roulant le code nécessaire pour la question f du devoir 2.
"""
import os
from src.solver.solver import first_order, second_order
from src.postprocessing.plotter import plotter, plotter_time

params = {
    "RI": 0,
    "RO": 0.5,
    "S": 2e-8,
    "D_EFF": 10e-10,
    "CE": 20,
    "K": 4e-9,
    "TF": 4e9
}

if __name__ == "__main__":
    # Création robuste du dossier results
    dossier_courant = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(dossier_courant, "results"), exist_ok=True)

    intervales_spatiaux = 11
    intervales_temporels = 100000

    # Ajout du paramètre de pas de temps (DT) au dictionnaire
    params["DT"] = params["TF"] / intervales_temporels

    # Appel du solveur numérique
    discretisation_second, tableau_temps_second, concentration_num_2d_second = second_order(
        params,
        intervales_spatiaux)
    # Appel du solveur numérique
    discretisation_first, tableau_temps_first, concentration_num_2d_first = first_order(
        params,
        intervales_spatiaux)

    plotter(params, discretisation_second, concentration_num_2d_second[-1],
            order="2", save_path="results/profil_concentration_second_order.png")
    plotter(params, discretisation_first, concentration_num_2d_first[-1],
            order="1", save_path="results/profil_concentration_first_order.png")
    
    plotter_time(params, discretisation_second, tableau_temps_second, concentration_num_2d_second,
             order="2", save_path="results/concentration_time_second_order.png")

    plotter_time(params, discretisation_first, tableau_temps_first, concentration_num_2d_first,
             order="1", save_path="results/concentration_time_first_order.png")
