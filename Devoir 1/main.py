"""
Fichier roulant le code nécessaire pour le devoir.
"""
import os
import numpy as np
from src.solver.solver import first_order, second_order, analytique
from src.postprocessing.plotter import (plotter, graph_error_log,
                                        graph_error_radius, print_convergence_table)
from src.verif.error import norm_l1, norm_l2, norm_infinity
from src.verif.symetrie import gen_symetrie

params = {
    "RI": 0,
    "RO": 0.5,
    "S": 2e-8,
    "D_EFF": 1e-10,
    "CE": 20
}

N_POINTS_MIN=10
N_POINTS_MAX=10000
NB_SIMULATION=5

if __name__ == "__main__":
    # Créer le dossier results s'il n'existe pas
    os.makedirs("results", exist_ok=True)

    list_n_points = np.logspace(
        np.log10(N_POINTS_MIN),
        np.log10(N_POINTS_MAX),
        NB_SIMULATION,
        dtype=int)

    dr_list = []
    l1_list_1, l2_list_1, linf_list_1 = [], [], []
    l1_list_2, l2_list_2, linf_list_2 = [], [], []

    for n_points in list_n_points:
        discretization_1, concentration_1 = first_order(params, n_points)
        discretization_2, concentration_2 = second_order(params,n_points)
        discretization_a, concentration_a = analytique(params,n_points)

        discretization = np.linspace(params["RI"], params["RO"], n_points)
        dr = discretization[1] - discretization[0]
        dr_list.append(dr)

        # First order norms
        l1_list_1.append(norm_l1(concentration_1, concentration_a))
        l2_list_1.append(norm_l2(concentration_1, concentration_a))
        linf_list_1.append(norm_infinity(concentration_1, concentration_a))

        # Second order norms
        l1_list_2.append(norm_l1(concentration_2, concentration_a))
        l2_list_2.append(norm_l2(concentration_2, concentration_a))
        linf_list_2.append(norm_infinity(concentration_2, concentration_a))

        # Plots first and last iterations
        if n_points in (list_n_points[0], list_n_points[-1]):
            plotter(params, discretization_1,
                    concentration_1, discretization_a,
                    concentration_a, order=1,
                    save_path=f"results/numeric_vs_analytic_order_1_npoints_{n_points}.png")

            plotter(params, discretization_2,
                    concentration_2, discretization_a,
                    concentration_a, order=2,
                    save_path=f"results/numeric_vs_analytic_order_2_npoints_{n_points}.png")
    
    # Vérification de symétrie
    gen_symetrie(params)
    
    # Plots
    graph_error_log(params, dr_list, l1_list_1, l2_list_1, linf_list_1,
                    order=1, save_path="results/error_log_order_1.png")

    graph_error_log(params, dr_list, l1_list_2, l2_list_2, linf_list_2,
                    order=2, save_path="results/error_log_order_2.png")

    graph_error_radius(params, discretization, concentration_1,
                       concentration_a, order=1,
                       save_path="results/error_radius_order_1.png")

    graph_error_radius(params, discretization, concentration_2,
                       concentration_a, order=2,
                       save_path="results/error_radius_order_2.png")

    # Convergence tables
    print_convergence_table(list_n_points, dr_list, l1_list_1,
                            order=1, label="L1")

    print_convergence_table(list_n_points, dr_list, l2_list_1,
                            order=1, label="L2")

    print_convergence_table(list_n_points, dr_list, linf_list_1,
                            order=1, label="LInf")

    print_convergence_table(list_n_points, dr_list, l1_list_2,
                            order=2, label="L1")

    print_convergence_table(list_n_points, dr_list, l2_list_2,
                            order=2, label="L2")

    print_convergence_table(list_n_points, dr_list, linf_list_2,
                            order=2, label="LInf")
