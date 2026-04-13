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
    compute_boundary_fluxes,
    compute_average_temperature,
    compute_temperature_at_y)
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


def calculer_gci_et_asymptotique(srq_list, p_observe, p_formel=2.0, ratio=2.0):
    """
    Calcule le GCI et vérifie la plage asymptotique pour les 3 maillages les plus fins.
    """
    f1 = srq_list[-1]  # Maillage fin
    f2 = srq_list[-2]  # Maillage moyen
    f3 = srq_list[-3]  # Maillage grossier

    ecart_relatif_p = abs((p_observe - p_formel) / p_formel)

    if ecart_relatif_p <= 0.1:
        Fs = 1.25
        p_utilise = p_formel
    else:
        Fs = 3.0
        p_utilise = min(max(0.5, p_observe), p_formel)

    # GCI = (Fs / (r^p - 1)) * |f2 - f1|
    gci_21_dim = (Fs / (ratio**p_utilise - 1)) * abs(f1 - f2)
    gci_32_dim = (Fs / (ratio**p_utilise - 1)) * abs(f2 - f3)

    gci_21_rel = (gci_21_dim / abs(f1)) * 100 if f1 != 0 else 0

    asymptotic_ratio = gci_32_dim / (ratio**p_utilise * gci_21_dim) if gci_21_dim != 0 else 0

    return gci_21_dim, gci_21_rel, asymptotic_ratio, Fs, p_utilise


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
    order_val = int(order)

    srq_heat_full, srq_heat_40, srq_temp_moy, srq_temp_pt = [], [], [], []

    for n in maille_list:
        print(f"Lancement des calculs sur le maillage de taille {n}x{n}...")
        local_dict['nx'], local_dict['ny'] = n, n

        if order_val == 1:
            temp = solver_first_order(local_dict)
        elif order_val == 2:
            temp = solver_second_order(local_dict, scheme)
        else:
            raise ValueError("L'ordre doit être 1 ou 2.")

        srq_heat_full.append(compute_boundary_fluxes(temp, local_dict, 0.0))
        srq_heat_40.append(compute_boundary_fluxes(temp, local_dict, 0.4))
        srq_temp_moy.append(compute_average_temperature(temp, local_dict))
        srq_temp_pt.append(compute_temperature_at_y(temp, local_dict, 0.8, 0.8))

    # --- CALCULS RICHARDSON & GCI ---
    srqs = {
        "Flux Total": srq_heat_full,
        "Flux Partiel": srq_heat_40,
        "Temp. Moyenne": srq_temp_moy,
        "Temp. Point (0.80, 0.80)": srq_temp_pt
    }

    print("\n" + "="*50)
    header = f"{'SRQ':<25} | {'p Obs.':<7} | {'p Util.':<7} | {'Fs':<4} | {'GCI (dim)':<11} | {'GCI (%)':<9} | {'Asympt.'}"
    print(header)
    print("-" * 50)

    for name, data in srqs.items():
        p_obs = calcul_ordre_convergence_richardson(data, maille_list, order_val)

        # Appel de la nouvelle fonction Roache
        res = calculer_gci_et_asymptotique(data, p_obs, p_formel=float(order_val))
        gci_dim, gci_rel, asymp, fs, p_used = res

        print(f"{name:<25} | {p_obs:<7.3f} | {p_used:<7.3f} | {fs:<4.2f} | {gci_dim:<11.3e} | {gci_rel:<9.3e} | {asymp:<8.3f}")

        nom_fichier_base = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').lower()

        plot_relative_error_loglog(
            data, maille_list, input_dict,
            f"Convergence : {name}",
            f"SOLUTION_VERIFICATION/loglog_{nom_fichier_base}.png"
            )
        plot_srq_vs_hp_with_gci(
            data, maille_list, p_used, gci_dim, gci_rel, input_dict,
            title=f"Convergence Asymptotique : {name}",
            filename=f"SOLUTION_VERIFICATION/asymptote_{nom_fichier_base}.png"
        )
    print("="*50 + "\n")


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


def plot_srq_vs_hp_with_gci(
        srq_list, maille_list, p_observe, gci_fin_dim, gci_fin_rel, input_dict,
        title="Convergence Asymptotique de la SRQ", filename="srq_vs_h.png"):
    """
    Trace la valeur absolue de la SRQ en fonction de h^p,
    avec l'extrapolation de Richardson et la barre d'erreur du GCI (dim et %).
    """
    srq_array = np.array(srq_list, dtype=float)
    n_array = np.array(maille_list, dtype=float)

    h_array = 1.0 / (n_array - 1.0)
    hp_array = h_array ** p_observe

    hp_fit = hp_array[-3:]
    srq_fit = srq_array[-3:]

    slope, intercept = np.polyfit(hp_fit, srq_fit, 1)
    f_exact = intercept

    plt.figure(figsize=(8, 6))

    hp_line = np.linspace(0, max(hp_array)*1.1, 100)
    plt.plot(hp_line, slope * hp_line + intercept, '--', color='gray',
             label=f"Extrapolation Richardson\n($f_{{exact}} \\approx {f_exact:.5g}$)")

    plt.plot(hp_array, srq_array, 's', markersize=8, color='#1f77b4', label='Solutions numériques calculées')

    label_gci = f"Incertitude GCI : ±{gci_fin_dim:.2e} ({gci_fin_rel:.2e} %)"
    plt.errorbar(hp_array[-1], srq_array[-1], yerr=gci_fin_dim, fmt='none', ecolor='red',
                 capsize=5, markeredgewidth=2, label=label_gci)

    plt.plot(0, f_exact, 'r*', markersize=14, label='Valeur Asymptotique (h=0)')

    plt.xlabel(f"Taille de maille à la puissance $p$ ($h^{{{p_observe:.2f}}}$)", fontsize=12)
    plt.ylabel("Valeur absolue de la SRQ", fontsize=12)
    plt.ticklabel_format(useOffset=False, axis='y')
    plt.title(title, fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=-max(hp_array)*0.05)

    plt.legend()
    plt.tight_layout()

    save_full_path = os.path.join(input_dict['save_path'], filename)
    plt.savefig(save_full_path, dpi=300)
    plt.close()
