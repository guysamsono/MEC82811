"""
Fonctions servant à afficher les résultats.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def graph_error_log(input_dict: dict,
                    discretization,
                    l1_list, l2_list, linf_list,
                    order,
                    variable_convergence,
                    file_name="convergence.png",
                    show_fig=False,
                    xlabel=r"Taille de maille"):
    """
    Affiche les erreurs L1, L2 et Linf en échelle log-log.

    Paramètres
    ----------
    input_dict : dict
        Dictionnaire des paramètres du problème.
    discretization : array-like
        Valeurs de discrétisation (ex: dx, dy, ou h).
    l1_list, l2_list, linf_list : array-like
        Normes d'erreur.
    order : str
        Nom ou ordre du schéma.
    variable_convergence : str
        Variable sur laquelle la convergence est étudiée
        (ex: 'x', 'y', 'x et y').
    save_path : str
        Chemin de sauvegarde.
    show_fig : bool
        Afficher la figure à l'écran ou non.
    xlabel : str
        Étiquette de l'axe x.
    """

    rho = input_dict['rho']
    cp = input_dict['cp']
    kappa = input_dict['k']
    f = input_dict['f']
    h = input_dict['h']

    plt.figure(figsize=(8, 6))

    plt.loglog(discretization, l1_list, 'o-', label=r"$L_1$")
    plt.loglog(discretization, l2_list, 's-', label=r"$L_2$")
    plt.loglog(discretization, linf_list, '^-', label=r"$L_\infty$")

    plt.xlabel(xlabel)
    plt.ylabel("Erreur")
    plt.title(
        f"Convergence de la solution numérique ({order})\n"
        f"Variable raffinée : {variable_convergence}\n"
        f"rho={rho}, cp={cp}, k={kappa}, f={f}, h={h}"
    )
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    save_full_path = os.path.join(input_dict['save_path'], file_name)
    plt.savefig(save_full_path, dpi=300)

    if show_fig:
        plt.show()
    plt.close()



# pylint: disable=too-many-locals
def print_convergence_table(n_points_list, discretization_list,
                            error_list, order,
                            label):
    '''
    Fonction qui print une table de convergence.

    :param n_points_list: liste des nombres de points
    :param dr_list: liste des discretisations
    :param error_list: liste des erreur
    :param label: label d'affichage
    '''

    dr = np.array(discretization_list)
    err = np.array(error_list)

    numerator = np.log(err[:-1] / err[1:])
    denominator = np.log(dr[:-1] / dr[1:])
    p_rates = numerator / denominator

    print(f"\n{'='*60}")
    print(f"ANALYSE DE CONVERGENCE DE L'ORDRE {order}: Norme {label}")
    print(f"{'='*60}")

    header = f"{'N points':^10} | {'dr [m]':^12} | {'Erreur':^12} | {'Ordre p':^10}"
    print(header)
    print(f"{'-'*11}|{'-'*14}|{'-'*14}|{'-'*12}")

    n_0 = n_points_list[0]
    d_0 = discretization_list[0]
    e_0 = error_list[0]
    print(f"{n_0:^10} | {d_0:^12.2e} | {e_0:^12.2e} | {'-':^10}")

    for i, p in enumerate(p_rates):
        n = n_points_list[i+1]
        d = discretization_list[i+1]
        e = error_list[i+1]

        print(f"{n:^10} | {d:^12.2e} | {e:^12.2e} | {p:^10.4f}")
    print(f"{'='*60}\n")
