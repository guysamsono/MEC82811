# Fichier: analyse_de_convergence.py
#
# But: Tracer un graphique d'analyse de convergence avec regression en loi de puissance à partir des 
#      données contenues dans les fichiers "erreurs" et liste_des_resolutions. Ce dernier nom et les variables
#      0 et 100.0 seront remplacés par le script bash analyse_auto qui appelle ce programme python.

import numpy as np
import matplotlib.pyplot as plt
import os

# Fonction pour lire des fichiers de données
def reading_files():

    # Lecture des erreurs absolues
    with open('src/bash/erreurs', 'r') as fichier1:
        error_values = [float(ligne) for ligne in fichier1.read().splitlines() if ligne.strip()]

    # Lecture de nombre d'intervalle utilisée pour l'intégration dans le fichier passé en 2eme argument du script bash
    with open('src/bash/liste_des_resolutions', 'r') as fichier2:
        lignes = fichier2.read().splitlines()
        nombres = [int(ligne.split()[0]) for ligne in lignes if ligne.strip()]

    # Calcul du pas d'intégration (Δx = h) en fonction des bornes qui se trouvemt dans le fichier trapezoidal.cpp
    h_values = [(0.5 - 0) / nombre for nombre in nombres] 

    # Trier `h_values` par ordre croissant et appliquer le même ordre à `error_values`
    paires_triees = sorted(zip(h_values, error_values))

    # Déballer les paires triées
    h_values_triees, error_values_triees = zip(*paires_triees)

    # Convertir les tuples en listes
    h_values = list(h_values_triees)
    error_values = list(error_values_triees)

    return h_values, error_values

# Lire les données
h_values, error_values = reading_files()

# Ajuster une loi de puissance sur les trois premières valeurs en utilisant np.polyfit avec logarithmes)
coefficients = np.polyfit(np.log(h_values[:2]), np.log(error_values[:2]), 1)
exponent = coefficients[0]

# Fonction de régression en termes de logarithmes
fit_function_log = lambda x: exponent * x + coefficients[1]

# Fonction de régression en termes originaux
fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_function(h_values[-1])

# Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
plt.figure(figsize=(8, 6))
plt.scatter(h_values, error_values, marker='o', color='b', label='Données numériques obtenues')
plt.plot(h_values, fit_function(h_values), linestyle='--', color='r', label='Régression en loi de puissance')

# Ajouter des étiquettes et un titre au graphique
plt.title("Convergence observée de l'équation de diffusion-réaction en fonction de $Δx$",
          fontsize=14, fontweight='bold', y=1.02)  # Le paramètre y règle la position verticale du titre

plt.xlabel('$Δx$', fontsize=12, fontweight='bold') 
plt.ylabel('Erreur $L_2$', fontsize=12, fontweight='bold')

# Rendre les axes plus gras
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)

# Placer les marques de coche à l'intérieur et les rendre un peu plus longues
plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)

# Afficher l'équation de la régression en loi de puissance
equation_text = f'$L_2 = {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')

# Déplacer la zone de texte
equation_text_obj.set_position((0.5, 0.4))


# Afficher le graphique
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig("results/analyse_convergence_automatique.png")
plt.show()
