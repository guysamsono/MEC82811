from src.solver import *
from src.error import norm_l1, norm_l2, norm_infinity
from src.convergence import graph_error_log, print_convergence_table
from src.mms import *
import os
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def calcul_ordre_convergence_richardson(srq_list, maille_list, p_init=2.0, tol=1e-10, max_iter=1000):
    """
    Calcule l'ordre de convergence observé p avec la méthode itérative de Roache.

    Paramètres
    ----------
    srq_list : list[float]
        Valeurs de la quantité d'intérêt sur différents maillages.
    maille_list : list[int]
        Nombre de points dans chaque direction pour chaque maillage.
    p_init : float, optionnel
        Estimation initiale de p.
    tol : float, optionnel
        Tolérance de convergence sur p.
    max_iter : int, optionnel
        Nombre maximal d'itérations.

    Retour
    ------
    p : float
        Ordre de convergence observé.
    """

    f1 = srq_list[-1]   # maillage le plus fin
    f2 = srq_list[-2]
    f3 = srq_list[-3]

    n1 = maille_list[-1]
    n2 = maille_list[-2]
    n3 = maille_list[-3]

    # Tailles de maille h ~ 1/(n-1)
    h1 = 1.0 / (n1 - 1)
    h2 = 1.0 / (n2 - 1)
    h3 = 1.0 / (n3 - 1)

    r12 = h2 / h1
    r23 = h3 / h2

    e32 = f3 - f2
    e21 = f2 - f1

    # print("f3, f2, f1 =", f3, f2, f1)
    # print("h3, h2, h1 =", h3, h2, h1)
    # print("r12 =", r12)
    # print("r23 =", r23)
    # print("e32 =", e32)
    # print("e21 =", e21)
    # print("ratio e32/e21 =", e32 / e21)

    if abs(e21) < 1e-14:
        raise ValueError("f2 - f1 est trop proche de 0, impossible de calculer l'ordre.")
    if r12 <= 0 or r23 <= 0:
        raise ValueError("r12 et r23 doivent être strictement positifs.")
    if abs(r12 - 1.0) < 1e-14 or abs(r23 - 1.0) < 1e-14:
        raise ValueError("r12 et r23 ne doivent pas être égaux à 1.")

    p = p_init

    for _ in range(max_iter):
        arg = ((r12**p - 1.0) * (e32 / e21)) + r12**p

        if arg <= 0:
            raise ValueError(
                "Argument du logarithme <= 0. Vérifie les maillages ou les valeurs de la SRQ."
            )

        p_new = np.log(arg) / np.log(r12 * r23)

        if abs(p_new - p) < tol:
            return p_new

        p = p_new

    raise RuntimeError("La méthode itérative n'a pas convergé.")

def calcul_p_hat(k_list, ratio):
    p_hat = (np.log((k_list[-3]-k_list[-2])/(k_list[-2]-k_list[-1])))/(np.log(ratio))
    return p_hat


def plot_relative_error_loglog(srq_list, maille_list, input_dict, title="Erreur relative sur la SRQ (%)", filename="srq_convergence.png"):
    """
    Trace l'erreur relative de la SRQ par rapport à la solution la plus fine
    sur un graphique log-log, et calcule la régression de type y = C x^p.

    Paramètres
    ----------
    srq_list : list of float
        Valeurs de la quantité d'intérêt pour chaque maillage.
    maille_list : list of int
        Nombre de points de maillage correspondants.
    title : str
        Titre du graphique.

    Retour
    ------
    slope : float
        Pente de la régression log-log.
    intercept : float
        Ordonnée à l'origine dans l'espace log.
    C : float
        Constante dans y = C x^p.
    r2 : float
        Coefficient de détermination de la régression.
    """

    srq_array = np.array(srq_list, dtype=float)
    n_array = np.array(maille_list, dtype=float)

    # Solution de référence = solution la plus fine
    srq_ref = srq_array[-1]

    # h ~ 1/(n-1)
    h_array = 1.0 / (n_array - 1.0)

    # On exclut le point le plus fin, car son erreur relative serait 0
    h_plot = h_array[:-1]
    srq_plot = srq_array[:-1]

    # Erreur relative en %
    rel_error = np.abs((srq_plot - srq_ref) / srq_ref) * 100.0

    # Régression dans l'espace log-log
    log_h = np.log10(h_plot)
    log_e = np.log10(rel_error)

    slope, intercept = np.polyfit(log_h, log_e, 1)

    log_e_fit = slope * log_h + intercept
    e_fit = 10**log_e_fit
    C = 10**intercept

    # Calcul de R² dans l'espace log-log
    ss_res = np.sum((log_e - log_e_fit)**2)
    ss_tot = np.sum((log_e - np.mean(log_e))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # Plot
    plt.figure(figsize=(7, 5))
    plt.loglog(h_plot, rel_error, 'D', markersize=7, label='Données')
    plt.loglog(h_plot, e_fit, '--', linewidth=1.5, label='Régression')

    plt.xlabel("Taille de maille h")
    plt.ylabel("Erreur relative sur la SRQ (%)")
    plt.title(title)
    plt.grid(True, which="major", linestyle='-', alpha=0.6)
    plt.grid(True, which="minor", linestyle=':', alpha=0.4)

    eq_text = f"$y = {C:.2e} x^{{{slope:.2f}}}$\n$R^2 = {r2:.2f}$"
    plt.text(0.58, 0.82, eq_text, transform=plt.gca().transAxes,
             color='red', fontsize=14)

    plt.legend()
    plt.tight_layout()

    save_full_path = os.path.join(input_dict['save_path'], filename)
    plt.savefig(save_full_path, dpi=300)
    plt.close()

    # print(f"Régression log-log : y = {C:.6e} * x^{slope:.6f}")
    # print(f"R² = {r2:.6f}")

    return slope, intercept, C, r2


def solution_verification(input_dict, order=2, scheme='central'):
    maille_list = [51, 101, 201, 401, 801]
    local_dict = input_dict.copy()
    srq_list_temperature_centrale, srq_list_temperature_max, srq_list_heat_transfer_full, srq_list_heat_transfer_40 = [], [], [], []
    srq_list_temperature_max = []

    for n in maille_list:
        local_dict['nx'] = n
        local_dict['ny'] = n

        if order == 1:
            temperature = solver_first_order(local_dict)
        elif order == 2:
            temperature = solver_second_order(local_dict, scheme)
        
        heat_transfer_0 = compute_boundary_fluxes(temperature, local_dict, margin_ratio=0.0)
        heat_transfer_40 = compute_boundary_fluxes(temperature, local_dict, margin_ratio=0.4)
        srq_list_heat_transfer_full.append(heat_transfer_0)
        srq_list_heat_transfer_40.append(heat_transfer_40)

        # Mid Temp SRQ
        i_mid = local_dict['ny'] // 2
        j_mid = local_dict['nx'] // 2
        k_mid = i_mid * local_dict['nx'] + j_mid
        srq_list_temperature_centrale.append(temperature[k_mid])
        srq_list_temperature_max.append(max(temperature))

    p_hat_rich_temp_centrale = calcul_ordre_convergence_richardson(srq_list_temperature_centrale, maille_list, p_init=order)
    p_hat_rich_temp_max = calcul_ordre_convergence_richardson(srq_list_temperature_max, maille_list, p_init=order)
    p_hat_rich_heat_transfer_full = calcul_ordre_convergence_richardson(srq_list_heat_transfer_full, maille_list, p_init=order)
    p_hat_rich_heat_transfer_central = calcul_ordre_convergence_richardson(srq_list_heat_transfer_40, maille_list, p_init=order)

    p_hat_temp_centrale = calcul_p_hat(srq_list_temperature_centrale, 2)
    p_hat_temp_max = calcul_p_hat(srq_list_temperature_max, 2)
    p_hat_heat_transfer_full = calcul_p_hat(srq_list_heat_transfer_full, 2)
    p_hat_heat_transfer_central = calcul_p_hat(srq_list_heat_transfer_40, 2)

    print(f"L'ordre de convergence pour la température centrale est {p_hat_temp_centrale}")
    print(f"L'ordre de convergence pour la température maximale est {p_hat_temp_max}")
    print(f"L'ordre de convergence pour le heat transfer total est {p_hat_heat_transfer_full}")
    print(f"L'ordre de convergence pour le heat transfer cental est {p_hat_heat_transfer_central}")
    
    plot_relative_error_loglog(srq_list_temperature_centrale, maille_list, input_dict, title="Erreur relative sur la température centrale (%)", filename="srq_convergence_temp_centrale.png")
    plot_relative_error_loglog(srq_list_temperature_max, maille_list, input_dict, title="Erreur relative sur la température maximale (%)", filename="srq_convergence_temp_max.png")
    plot_relative_error_loglog(srq_list_heat_transfer_full, maille_list, input_dict, title="Erreur relative sur le transfer de chaleur total (%)", filename="srq_convergence_heat_full.png")
    plot_relative_error_loglog(srq_list_heat_transfer_40, maille_list, input_dict, title="Erreur relative sur le transfer de chaleur partiel (%)", filename="srq_convergence_heat_center.png")

def post_processing_verification(input_dict):
    maille_list = [100,200,400,500,600,700,800,900,1000]
    local_dict = input_dict.copy()
    f_T_MMS, f_source, f_bc_left, f_bc_right, f_bc_bottom, f_tinf_top = generer_mms_simple(local_dict, afficher_graphiques=False)

    srq_list = []

    for n in maille_list:
        local_dict['nx'] = n
        local_dict['ny'] = n

        temperature_exact = mms_Temperature(local_dict, f_T_MMS)
        heat_transfer = compute_boundary_fluxes(temperature_exact,local_dict)
        # energy_conservation = compute_conservation_of_energy(temperature_exact,input_dict,f_source=f_source,
        #                                                       bc_top_tinf=f_tinf_top,bc_bottom=f_bc_bottom,order=2)

        srq_list.append(heat_transfer)

    p_hat_rich = calcul_ordre_convergence_richardson(srq_list, maille_list, p_init=2)

    p_hat = calcul_p_hat(srq_list,2)

    print(p_hat_rich)
    
    plot_relative_error_loglog(srq_list, maille_list, input_dict)

    print(srq_list)


    
