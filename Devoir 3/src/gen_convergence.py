import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from lbm_accel import Generate_sample, LBM

def calcul_p_hat(k_list, ratio):
    p_hat = (np.log((k_list[0]-k_list[1])/(k_list[1]-k_list[2])))/(np.log(ratio))
    return p_hat

def gci_calculation(p_hat, p_f, k_list, nx_list):
    """
    Calcule le Grid Convergence Index (GCI) entre les deux maillages les plus fins.

    Parameters
    ----------
    p_hat : float
        Ordre de convergence observé.
    p_f : float
        Ordre formel attendu.
    k_list : array-like
        Liste des solutions, du plus grossier au plus fin.
    nx_list : array-like
        Liste des tailles de maillage correspondantes, du plus grossier au plus fin.

    Returns
    -------
    GCI : float
        Grid Convergence Index.
    """

    k_list = np.asarray(k_list, dtype=float)
    nx_list = np.asarray(nx_list, dtype=float)

    if len(k_list) < 2 or len(nx_list) < 2:
        raise ValueError("k_list et nx_list doivent contenir au moins 2 valeurs.")

    if len(k_list) != len(nx_list):
        raise ValueError("k_list et nx_list doivent avoir la même longueur.")

    # Rapport de raffinement entre les deux maillages les plus fins
    r = nx_list[-1] / nx_list[-2]

    if r <= 1:
        raise ValueError("Le ratio de raffinement r doit être > 1.")

    rapport = abs((p_hat - p_f) / p_f)

    if rapport <= 0.1:
        Fs = 1.25
        P = p_f
    else:
        Fs = 3.0
        P = min(max(0.5, p_hat), p_f)

    GCI = Fs / (r**P - 1) * abs(k_list[-1] - k_list[-2])

    return GCI


def calcul_ordre_convergence_richardson(f,x, p_init=2.0, tol=1e-10, max_iter=1000):
    """
    Calcule l'ordre de convergence observé p avec la méthode itérative de Roache.

    Paramètres
    ----------
    f1 : float
        Solution sur le maillage le plus fin.
    f2 : float
        Solution sur le maillage intermédiaire.
    f3 : float
        Solution sur le maillage le plus grossier.
    r12 : float
        Ratio de raffinement entre les maillages 1 et 2.
        En général r12 = h2 / h1.
    r23 : float
        Ratio de raffinement entre les maillages 2 et 3.
        En général r23 = h3 / h2.
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

    f1 = f[-1]
    f2 = f[-2]
    f3 = f[-3]
    r12 = x[-2]/x[-1]
    r23 = x[-3]/x[-2]

    if f2 == f1:
        raise ValueError("f2 - f1 = 0, impossible de calculer l'ordre.")
    if r12 <= 0 or r23 <= 0:
        raise ValueError("r12 et r23 doivent être strictement positifs.")
    if r12 == 1 or r23 == 1:
        raise ValueError("r12 et r23 ne doivent pas être égaux à 1.")

    p = p_init

    for _ in range(max_iter):
        arg = ((r12**p - 1.0) * (f3 - f2) / (f2 - f1)) + r12**p

        if arg <= 0:
            raise ValueError(
                "Argument du logarithme <= 0. Vérifie l'ordre des maillages, "
                "les ratios r12/r23, ou les valeurs f1, f2, f3."
            )

        p_new = np.log(arg) / np.log(r12 * r23)

        if abs(p_new - p) < tol:
            return p_new

        p = p_new

    print("f3, f2, f1 =", f3, f2, f1)
    print("e32 =", f3 - f2)
    print("e21 =", f2 - f1)
    print("ratio e32/e21 =", (f3 - f2)/(f2 - f1))

    raise RuntimeError("La méthode itérative n'a pas convergé.")
    

def gen_convergence_func(ratio, deltaP, NX, poro, mean_fiber_d, std_d, dx, filename, seed=101):
    
    nx_list = [NX/ratio, NX, NX*ratio]
    dx_list = [dx*ratio, dx, dx/ratio]
    k_list = []

    for i in range(len(nx_list)):
        print(f"Running LBM with NX={nx_list[i]} and dx={dx_list[i]}")
        d_equivalent = Generate_sample(seed, filename, mean_fiber_d, std_d, poro, int(nx_list[i]), dx_list[i])
        k_list.append(LBM(filename, int(nx_list[i]), deltaP, dx_list[i], d_equivalent))

    # Plotting the results
    plt.figure(figsize=(8, 6))
    plt.loglog(dx_list, k_list, marker='o', label='LBM Permeability')
    plt.xlabel('Grid Spacing (dx)')
    plt.ylabel('Permeability (k)')
    plt.title('Convergence of LBM Permeability with Grid Refinement')
    plt.grid(True, which="both", ls="--")
    try:
        plt.savefig('results/convergence_plot.png')
    except Exception as e:
        print(f"Error saving plot: {e}, does the 'results' directory exist?")
    plt.show()

    p_hat = calcul_p_hat(k_list, ratio)
    print(f"Estimated order of convergence (p_hat): {p_hat:.2f}")
    GCI = gci_calculation(p_hat, 2,k_list,ratio)
    print(f"Estimated Grid Convergence Index: {GCI:.2f}")

def gen_convergence_mean_func(deltaP,nx_list,dx_list,seed_list,poro,mean_fiber_d,std_d,filename):
    import os
    dossier_courant = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(dossier_courant, "results"), exist_ok=True)

    k_mean_list = []
    k_std_list = []
    k_all_by_nx = []

    for i in range(len(nx_list)):
        nx = nx_list[i]
        dx = dx_list[i]

        k_values_for_this_nx = []

        for seed in seed_list:
                print(f"Running LBM with NX={nx_list[i]}, dx={dx_list[i]}, and seed={seed}")
                
                d_equivalent = Generate_sample(seed, filename, mean_fiber_d, std_d, poro, int(nx), dx)
                k_values_for_this_nx.append(LBM(filename, int(nx), deltaP, dx, d_equivalent))
        
        k_values_for_this_nx = np.array(k_values_for_this_nx)
        k_mean = np.mean(k_values_for_this_nx)
        k_std = np.std(k_values_for_this_nx,ddof=1)

        k_mean_list.append(k_mean)
        k_std_list.append(k_std)
        k_all_by_nx.append(k_values_for_this_nx)

    # Graphe 1 : convergence log-log
    k_ref = k_mean_list[-1]
    rel_error = np.abs((k_mean_list - k_ref) / k_ref)
    dx_array = np.array(dx_list, dtype=float)

    x = (dx_array / 1e-6)[:-1]   # dx en µm
    y = (rel_error * 100)[:-1]   # erreur en %

    logx = np.log10(x)
    logy = np.log10(y)

    slope, intercept = np.polyfit(logx, logy, 1)
    logy_fit = slope * logx + intercept
    y_fit = 10**logy_fit

    ss_res = np.sum((logy - logy_fit)**2)
    ss_tot = np.sum((logy - np.mean(logy))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    equation_text = (
        f"y = {10**intercept:.3e} x^{slope:.3f}\n"
        f"R² = {r2:.4f}"
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(x, y, 'o', label="Données")
    ax.loglog(x, y_fit, '--', label="Régression linéaire")

    ax.set_xlabel("dx (μm)")
    ax.set_ylabel("Erreur relative sur la perméabilité moyenne")
    ax.set_title("Convergence de la perméabilité moyenne")

    # Affichage de l'axe y en pourcentage
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))

    ax.grid(True, which="both")
    ax.legend()

    ax.text(
        0.05, 0.05, equation_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig("results/convergence_plot_error.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Graphe 2 : moyenne + incertitude
    plt.figure(figsize=(7, 5))
    plt.errorbar(dx_array / 1e-6, k_mean_list, yerr=k_std_list, fmt='o-', capsize=5)
    plt.xlabel("dx (μm)")
    plt.ylabel("Perméabilité moyenne (μm²)")
    plt.title("Perméabilité moyenne en fonction du maillage")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/moyenne_incertitude.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    ordre_convergence_reg = slope
    ordre_convergence_richardson = calcul_ordre_convergence_richardson(k_mean_list, dx_list)

    print(f"Ordre de convergence estimé à partir de la régression : {ordre_convergence_reg:.3f}")
    print(f"Ordre de convergence estimé à partir de la méthode de Richardson : {ordre_convergence_richardson:.3f}")

    # Graphe 3 : k moyen vs dx
    plt.figure(figsize=(8, 6))
    plt.loglog(dx_array / 1e-6, k_mean_list, marker='o', label='Perméabilité moyenne')
    plt.xlabel('Taille des cellules (dx) [μm]')
    plt.ylabel('Perméabilité moyenne (μm²)')
    plt.title('Convergence de la perméabilité moyenne avec le raffinement du maillage')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/convergence_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


    p_hat = ordre_convergence_reg
    print(f"Estimated order of convergence (p_hat): {p_hat:.2f}")
    GCI = gci_calculation(p_hat, 2,k_mean_list,nx_list)
    print(f"Estimated Grid Convergence Index: {GCI:.2f}")

    return GCI,p_hat

def plot_domain(deltaP, nx_list, dx_list, seed_list, poro, mean_fiber_d, std_d, filename_base):
    seed = seed_list[-1]

    dossier_courant = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(dossier_courant, "results"), exist_ok=True)

    for i, (nx, dx) in enumerate(zip(nx_list, dx_list)):
        dx_um = dx * 1e6
        domain_um = nx * dx_um

        tiff_name = f"results/{filename_base}_nx{nx}_dx{dx_um:.2f}um.tiff"

        d_equivalent = Generate_sample(
            seed=seed,
            filename=tiff_name,
            mean_d=mean_fiber_d,
            std_d=std_d,
            poro=poro,
            nx=int(nx),
            dx=dx,
            plot=True
        )

        fig = plt.gcf()
        ax = plt.gca()

        title = (
            f"Structure générée\n"
            f"nx = {nx}, dx = {dx_um:.2f} µm, L = {domain_um:.2f} µm\n"
            f"Porosité cible = {poro:.3f}, d_eq = {d_equivalent:.3f} µm"
        )
        ax.set_title(title)

        png_name = f"results/{filename_base}_nx{nx}_dx{dx_um:.2f}um.png"
        fig.savefig(png_name, dpi=300, bbox_inches="tight")

        plt.close(fig)
            