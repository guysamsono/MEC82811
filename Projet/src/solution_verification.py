"""
Module de vérification de solution.

Contient les fonctions pour évaluer l'ordre de convergence observé
via la méthode itérative de rich et l'extrapolation de Richardson.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from src.solver import (
    solver_first_order,
    solver_second_order,
    compute_boundary_fluxes)
from src.mms import generer_mms_simple, mms_temperature

def calcul_ordre_convergence_richardson(
        srq_list, maille_list, p_init=2.0,
        tol=1e-10, max_iter=1000):
    """
    Calcule l'ordre de convergence observé p avec la méthode itérative de rich.

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
    # pylint: disable=too-many-locals
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
    """
    Calcule l'ordre de convergence observé avec un ratio de raffinement constant.
    """
    p_hat = np.log((k_list[-3] - k_list[-2]) / (k_list[-2] - k_list[-1])) / np.log(ratio)
    return p_hat


def plot_relative_error_loglog(
        srq_list, maille_list, input_dict,
        title="Erreur relative sur la SRQ (%)", filename="srq_convergence.png"
):
    """
    Trace l'erreur relative de la SRQ par rapport à la solution la plus fine.
    """
    # pylint: disable=too-many-locals
    srq_array = np.array(srq_list, dtype=float)
    n_array = np.array(maille_list, dtype=float)

    srq_ref = srq_array[-1]
    h_array = 1.0 / (n_array - 1.0)

    h_plot = h_array[:-1]
    srq_plot = srq_array[:-1]

    rel_error = np.abs((srq_plot - srq_ref) / srq_ref) * 100.0

    log_h = np.log10(h_plot)
    log_e = np.log10(rel_error)

    slope, intercept = np.polyfit(log_h, log_e, 1)

    log_e_fit = slope * log_h + intercept
    e_fit = 10**log_e_fit
    c_const = 10**intercept

    ss_res = np.sum((log_e - log_e_fit)**2)
    ss_tot = np.sum((log_e - np.mean(log_e))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    plt.figure(figsize=(7, 5))
    plt.loglog(h_plot, rel_error, 'D', markersize=7, label='Données')
    plt.loglog(h_plot, e_fit, '--', linewidth=1.5, label='Régression')

    plt.xlabel("Taille de maille h")
    plt.ylabel("Erreur relative sur la SRQ (%)")
    plt.title(title)
    plt.grid(True, which="major", linestyle='-', alpha=0.6)
    plt.grid(True, which="minor", linestyle=':', alpha=0.4)

    eq_text = f"$y = {c_const:.2e} x^{{{slope:.2f}}}$\n$R^2 = {r2:.2f}$"
    plt.text(
        0.58, 0.82, eq_text, transform=plt.gca().transAxes,
        color='red', fontsize=14
    )

    plt.legend()
    plt.tight_layout()

    save_full_path = os.path.join(input_dict['save_path'], filename)
    plt.savefig(save_full_path, dpi=300)
    plt.close()

    return slope, intercept, c_const, r2


def solution_verification(input_dict, order=2, scheme='central'):
    """
    Gère la boucle de vérification de solution et d'extrapolation
    sur une liste de maillages prédéfinie.
    """
    # pylint: disable=too-many-locals

    sol_verif_dir = os.path.join(input_dict['save_path'], 'SOLUTION_VERIFICATION')
    os.makedirs(sol_verif_dir, exist_ok=True)

    maille_list = [51, 101, 201, 401, 801]
    local_dict = input_dict.copy()

    srq_list_temp_centrale = []
    srq_list_temp_max = []
    srq_list_heat_full = []
    srq_list_heat_40 = []

    for n in maille_list:
        local_dict['nx'] = n
        local_dict['ny'] = n

        if int(order) == 1:
            temperature = solver_first_order(local_dict)
        elif int(order) == 2:
            temperature = solver_second_order(local_dict, scheme)
        else:
            raise ValueError("L'ordre doit être 1 ou 2.")

        heat_transfer_0 = compute_boundary_fluxes(temperature, local_dict, margin_ratio=0.0)
        heat_transfer_40 = compute_boundary_fluxes(temperature, local_dict, margin_ratio=0.4)

        srq_list_heat_full.append(heat_transfer_0)
        srq_list_heat_40.append(heat_transfer_40)

        i_mid = local_dict['ny'] // 2
        j_mid = local_dict['nx'] // 2
        k_mid = i_mid * local_dict['nx'] + j_mid

        srq_list_temp_centrale.append(temperature[k_mid])
        srq_list_temp_max.append(max(temperature))

    p_rich_tc = calcul_ordre_convergence_richardson(srq_list_temp_centrale, maille_list, order)
    p_rich_tm = calcul_ordre_convergence_richardson(srq_list_temp_max, maille_list, order)
    p_rich_hf = calcul_ordre_convergence_richardson(srq_list_heat_full, maille_list, order)
    p_rich_hc = calcul_ordre_convergence_richardson(srq_list_heat_40, maille_list, order)

    p_hat_temp_centrale = calcul_p_hat(srq_list_temp_centrale, 2)
    p_hat_temp_max = calcul_p_hat(srq_list_temp_max, 2)
    p_hat_heat_full = calcul_p_hat(srq_list_heat_full, 2)
    p_hat_heat_central = calcul_p_hat(srq_list_heat_40, 2)

    print("\n--- Analyse de Vérification de Solution ---")
    print(
        f"Ordre central (Richardson): {p_rich_tc:.3f} | "
        f"Formule simple: {p_hat_temp_centrale:.3f}"
    )
    print(
        f"Ordre max (Richardson): {p_rich_tm:.3f} | "
        f"Formule simple: {p_hat_temp_max:.3f}"
    )
    print(
        f"Ordre flux total (Richardson): {p_rich_hf:.3f} | "
        f"Formule simple: {p_hat_heat_full:.3f}"
    )
    print(
        f"Ordre flux partiel (Richardson): {p_rich_hc:.3f} | "
        f"Formule simple: {p_hat_heat_central:.3f}"
    )

    plot_relative_error_loglog(
        srq_list_temp_centrale, maille_list, input_dict,
        title="Erreur relative sur la température centrale (%)",
        filename="SOLUTION_VERIFICATION/srq_convergence_temp_centrale.png"
    )
    plot_relative_error_loglog(
        srq_list_temp_max, maille_list, input_dict,
        title="Erreur relative sur la température maximale (%)",
        filename="SOLUTION_VERIFICATION/srq_convergence_temp_max.png"
    )
    plot_relative_error_loglog(
        srq_list_heat_full, maille_list, input_dict,
        title="Erreur relative sur le transfert de chaleur total (%)",
        filename="SOLUTION_VERIFICATION/srq_convergence_heat_full.png"
    )
    plot_relative_error_loglog(
        srq_list_heat_40, maille_list, input_dict,
        title="Erreur relative sur le transfert de chaleur partiel (%)",
        filename="SOLUTION_VERIFICATION/srq_convergence_heat_center.png"
    )


def post_processing_verification(input_dict):
    """
    Effectue une passe de post-traitement sur de gros maillages pour la MMS.
    """
    post_proc_dir = os.path.join(input_dict['save_path'], 'POST_PROCESSING')
    os.makedirs(post_proc_dir, exist_ok=True)

    maille_list = [100, 200, 400, 800, 1600]
    local_dict = input_dict.copy()

    f_t_mms, *_ = generer_mms_simple(local_dict)

    srq_list = []

    for n in maille_list:
        local_dict['nx'] = n
        local_dict['ny'] = n

        temperature_exact = mms_temperature(local_dict, f_t_mms)
        heat_transfer = compute_boundary_fluxes(temperature_exact, local_dict, 0.0)

        srq_list.append(heat_transfer)

    p_hat_rich = calcul_ordre_convergence_richardson(srq_list, maille_list, p_init=2)
    p_hat = calcul_p_hat(srq_list, 2)

    print(f"Ordre post-processing rich: {p_hat_rich}")
    print(f"Ordre post-processing simple: {p_hat}")

    plot_relative_error_loglog(
        srq_list, maille_list, input_dict,
        filename="POST_PROCESSING/post_processing_verification.png"
    )
