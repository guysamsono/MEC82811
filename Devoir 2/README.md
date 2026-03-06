# MEC8211 - Devoir 2 : Résolution numérique de l'équation de diffusion

Ce dépôt contient le code source pour le premier devoir du cours MEC8211 - Vérification et validation en modélisation numérique.

L'objectif est de résoudre l'équation de diffusion en coordonnées cylindriques (1D) pour prédire la concentration de chlorures dans une colonne de béton, en utilisant la méthode des différences finies (FDM) avec deux schémas numériques (Ordre 1 et Ordre 2).

## 📂 Structure du projet

```text
.
├── main.py           # Script principal (lance les simulations et génère les résultats)
├── src/
│   ├── solver/       # Contient les algorithmes de résolution (FDM Ordre 1 & 2)
│   ├── verif/        # Modules de calcul d'erreur et test de symétrie
│   └── postprocessing/ # Outils de génération de graphiques
├── results/          # Dossier de sortie (généré automatiquement)
├── .gitignore        # Fichiers ignorés par Git
└── README.md         # Documentation du projet
