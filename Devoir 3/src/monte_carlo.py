import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from lbm_accel import Generate_sample, LBM
import os

def monte_carlo_func(deltaP, n_runs=100,
                     mean_fiber_d_mean=12.5, mean_fiber_d_std=2.85,
                     poro_mean=0.900, poro_std=7.50e-3,
                     std_d_fixed=2.85,
                     NX=200, dx=2e-6,
                     filename="fiber_mat.tiff"):

    permeabilities = []
    sampled_mean_d = []
    sampled_poro = []

    rng = np.random.default_rng(42)

    for i in range(n_runs):
        mean_fiber_d_i = rng.normal(mean_fiber_d_mean, mean_fiber_d_std)
        poro_i = rng.normal(poro_mean, poro_std)

        while mean_fiber_d_i <= 0:
            mean_fiber_d_i = rng.normal(mean_fiber_d_mean, mean_fiber_d_std)

        while not (0 < poro_i < 1):
            poro_i = rng.normal(poro_mean, poro_std)

        # Geometry generation
        d_equivalent = Generate_sample(
            0,
            filename,
            mean_fiber_d_i,
            std_d_fixed,
            poro_i,
            int(NX),
            dx
        )

        # LBM solve
         # Calcul LBM
        k_i = LBM(filename, int(NX), deltaP, dx, d_equivalent)

        # Sécurité : lognormale exige k > 0
        if k_i is not None and np.isfinite(k_i) and k_i > 0:
            permeabilities.append(k_i)
            sampled_mean_d.append(mean_fiber_d_i)
            sampled_poro.append(poro_i)

    permeabilities = np.array(permeabilities, dtype=float)

    perm_log = np.log(permeabilities)
    # =========================
    k_mean = np.mean(perm_log)
    k_std = np.std(perm_log, ddof=1)

    # =========================
    # Fit log-normal
    # =========================
    # On force loc=0 dans la plupart des cas physiques
    shape, loc, scale = stats.lognorm.fit(permeabilities, floc=0)

    # Bornes multiplicatives géométriques
    k_geom_mean = np.exp(k_mean)              # médiane = moyenne géométrique
    k_minus_1sigma_log = np.exp(k_mean - k_std)
    k_plus_1sigma_log = np.exp(k_mean + k_std)

    # =========================
    # Vecteurs PDF / CDF
    # =========================
    x_min = permeabilities.min() * 0.90
    x_max = permeabilities.max() * 1.10
    pdf_x = np.linspace(x_min, x_max, 500)
    pdf_y = stats.lognorm.pdf(pdf_x, s=k_std, loc=loc, scale=scale)

    cdf_x = np.linspace(x_min, x_max, 500)
    cdf_y = stats.lognorm.cdf(cdf_x, s=k_std, loc=loc, scale=scale)

    # ECDF
    ecdf_x = np.sort(permeabilities)
    ecdf_y = np.arange(1, len(ecdf_x) + 1) / len(ecdf_x)

    # =========================
    # Affichage console
    # =========================
    print("\n===== Monte Carlo permeability results =====")
    print(f"Nombre d'échantillons valides = {len(permeabilities)}")
    print(f"Moyenne empirique            = {k_mean:.6e}")
    print(f"Écart-type empirique         = {k_std:.6e}")
    print(f"CV empirique (%)             = {100 * k_std / k_mean:.2f}")

    print("\n===== Log-normal fit =====")
    print(f"Moyenne lognormale           = {k_mean:.6e}")
    print(f"Écart-type lognormal         = {k_std:.6e}")
    print(f"exp(mu_log - sigma_log)      = {k_minus_1sigma_log:.6e}")
    print(f"exp(mu_log)                  = {k_geom_mean:.6e}")
    print(f"exp(mu_log + sigma_log)      = {k_plus_1sigma_log:.6e}")

    # =========================
    # Graphe PDF
    # =========================
    plt.figure(figsize=(8, 5))
    plt.hist(permeabilities, bins=20, density=True, alpha=0.6, label="Histogramme Monte Carlo")
    plt.plot(pdf_x, pdf_y, linewidth=2, label="Fit PDF log-normal")

    # Lignes verticales
    plt.axvline(k_minus_1sigma_log, linestyle=':', linewidth=2, label=f"exp(μlog-σlog) = {k_minus_1sigma_log:.3e}")
    plt.axvline(k_geom_mean, linestyle='-', linewidth=2, label=f"exp(μlog) = {k_geom_mean:.3e}")
    plt.axvline(k_plus_1sigma_log, linestyle=':', linewidth=2, label=f"exp(μlog+σlog) = {k_plus_1sigma_log:.3e}")

    plt.xlabel("Perméabilité")
    plt.ylabel("PDF")
    plt.title("PDF de la perméabilité - ajustement log-normal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig('results/pdf_monte_carlo.png')
    except FileNotFoundError:
        print('le fichier results/ sera crée')
        os.mkdir('results/')
        plt.savefig('results/pdf_monte_carlo.png')
    
    plt.figure(figsize=(8, 5))
    plt.step(ecdf_x, ecdf_y, where='post', label="CDF empirique")
    plt.plot(cdf_x, cdf_y, linewidth=2, label="Fit CDF log-normal")

    # Lignes verticales
    plt.axvline(k_minus_1sigma_log, linestyle=':', linewidth=2, label=f"exp(μlog-σlog) = {k_minus_1sigma_log:.3e}")
    plt.axvline(k_geom_mean, linestyle='-', linewidth=2, label=f"exp(μlog) = {k_geom_mean:.3e}")
    plt.axvline(k_plus_1sigma_log, linestyle=':', linewidth=2, label=f"exp(μlog+σlog) = {k_plus_1sigma_log:.3e}")

    plt.xlabel("Perméabilité")
    plt.ylabel("CDF")
    plt.title("CDF de la perméabilité - ajustement log-normal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig('results/cdf_monte_carlo.png')
    except FileNotFoundError:
        print('le fichier results/ sera crée')
        os.mkdir('results/')
        plt.savefig('results/cdf_monte_carlo.png')

    FVG = (pdf_x, pdf_y)
    CVG = (cdf_x, cdf_y)

    return {
        "permeabilities": permeabilities,
        "sampled_mean_d": sampled_mean_d,
        "sampled_poro": sampled_poro,

        "moyenne": k_mean,
        "écart-type": k_std,

        "k_minus_1sigma_log": k_minus_1sigma_log,
        "k_geom_mean": k_geom_mean,
        "k_plus_1sigma_log": k_plus_1sigma_log,

        "pdf_x": pdf_x,
        "pdf_y": pdf_y,
        "cdf_x": cdf_x,
        "cdf_y": cdf_y,
        "ecdf_x": ecdf_x,
        "ecdf_y": ecdf_y,

        "FVG": FVG,
        "CVG": CVG,
    }