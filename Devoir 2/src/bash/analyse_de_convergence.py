"""
Fichier traçant un graphique d'analyse de convergence à partir des données contenues dans les fichiers "erreurs" et
"liste_des_resolutions".
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour lire des fichiers de données
def reading_files():
    # Chargement des 3 normes dans un dictionnaire
    norms = {}
    for n in ['l1', 'l2', 'inf']:
        with open(f'src/bash/erreurs_{n}', 'r') as f:
            norms[n] = [float(l) for l in f.read().splitlines() if l.strip()]

    # Lecture des résolutions
    with open('src/bash/liste_des_resolutions', 'r') as f:
        lignes = f.read().splitlines()
        nr_list = [int(l.split()[0]) for l in lignes if l.strip()]
        nt_list = [int(l.split()[1]) for l in lignes if l.strip()]

    return nr_list, nt_list, norms

nr_list, nt_list, norms = reading_files()

# Paramètres du domaine physique
RAYON = 0.5
TEMPS_FINAL = 100.0

# Création du dossier de résultats
os.makedirs("results", exist_ok=True)

# Dictionnaire de configuration pour l'esthétique des 3 courbes
configs_normes = {
    'l1':  {'couleur': 'blue',   'marqueur': 'o', 'nom': '$L_1$'},
    'l2':  {'couleur': 'green',  'marqueur': 's', 'nom': '$L_2$'},
    'inf': {'couleur': 'orange', 'marqueur': '^', 'nom': r'$L_\infty$'}
}

# =========================================================
# ANALYSE SPATIALE
# =========================================================
max_nt = max(nt_list)
indices_spatiaux = [i for i, nt in enumerate(nt_list) if nt == max_nt]

# Si on a testé plusieurs nr pour ce temps gelé, on trace le graphique :
if len(set([nr_list[i] for i in indices_spatiaux])) > 1:
    h_values_raw = [(RAYON - 0) / nr_list[i] for i in indices_spatiaux]

    plt.figure(figsize=(8, 6))

    # Boucle sur les 3 normes
    for cle, config in configs_normes.items():
        e_values_raw = [norms[cle][i] for i in indices_spatiaux]

        # Tri croissant pour la régression
        h_values, e_values = zip(*sorted(zip(h_values_raw, e_values_raw)))

        # Régression linéaire sur le log
        coeffs = np.polyfit(np.log(h_values), np.log(e_values), 1)
        exp_spatial = coeffs[0]

        # Tracé des points et de la ligne
        label_texte = f"{config['nom']} (p={exp_spatial:.4f})"
        plt.scatter(h_values, e_values, marker=config['marqueur'], color=config['couleur'], s=60)
        plt.plot(h_values, np.exp(coeffs[1]) * np.array(h_values)**exp_spatial, '--', color=config['couleur'], linewidth=2, label=label_texte)

    plt.title(f"Convergence Spatiale ($N_t$={max_nt})", fontsize=14, fontweight='bold', y=1.02)
    plt.xlabel(r'$\Delta x$ (Pas spatial)', fontsize=12, fontweight='bold')
    plt.ylabel('Erreur', fontsize=12, fontweight='bold')

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)

    for spine in plt.gca().spines.values(): spine.set_linewidth(2)
    plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)

    plt.savefig("results/convergence_spatiale.png", dpi=300)
    print("--> Graphique spatial généré : results/convergence_spatiale.png")

# =========================================================
# ANALYSE TEMPORELLE
# =========================================================
max_nr = max(nr_list)
indices_temporels = [i for i, nr in enumerate(nr_list) if nr == max_nr]

# Si on a testé plusieurs nt pour cet espace gelé, on trace le graphique :
if len(set([nt_list[i] for i in indices_temporels])) > 1:
    dt_values_raw = [TEMPS_FINAL / nt_list[i] for i in indices_temporels]

    plt.figure(figsize=(8, 6))

    # Boucle sur les 3 normes
    for cle, config in configs_normes.items():
        e_values_raw = [norms[cle][i] for i in indices_temporels]

        # Tri croissant pour la régression
        dt_values, e_values = zip(*sorted(zip(dt_values_raw, e_values_raw)))

        # Régression linéaire sur le log
        coeffs = np.polyfit(np.log(dt_values), np.log(e_values), 1)
        exp_temporel = coeffs[0]

        # Tracé des points et de la ligne
        label_texte = f"{config['nom']} (p={exp_temporel:.4f})"
        plt.scatter(dt_values, e_values, marker=config['marqueur'], color=config['couleur'], s=60)
        plt.plot(dt_values, np.exp(coeffs[1]) * np.array(dt_values)**exp_temporel, '--', color=config['couleur'], linewidth=2, label=label_texte)

    plt.title(f"Convergence Temporelle ($N_r$={max_nr})", fontsize=14, fontweight='bold', y=1.02)
    plt.xlabel(r'$\Delta t$ (Pas de temps)', fontsize=12, fontweight='bold')
    plt.ylabel('Erreur', fontsize=12, fontweight='bold')

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)

    for spine in plt.gca().spines.values(): spine.set_linewidth(2)
    plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)

    plt.savefig("results/convergence_temporelle.png", dpi=300)
    print("--> Graphique temporel généré : results/convergence_temporelle.png")
