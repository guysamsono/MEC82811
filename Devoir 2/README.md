# MEC8211 - Devoir 2 : Résolution numérique de l'équation de diffusion

Ce dépôt contient le code source pour le premier devoir du cours MEC8211 - Vérification et validation en modélisation numérique.

L'objectif est de résoudre l'équation de diffusion en coordonnées cylindriques (1D) pour prédire la concentration de chlorures dans une colonne de béton, en utilisant la méthode des différences finies (FDM) avec deux schémas numériques (Ordre 1 et Ordre 2).

## 📂 Structure du projet

```text
.
├── README.md           # Documentation du projet
├── main.py             # Script principal de configuration et de simulation
├── results/            # Dossier de sortie (généré automatiquement par les scripts)
└── src/                # Code source du projet
    ├── bash/           # Scripts d'automatisation Bash et fichiers de données (résolutions)
    ├── postprocessing/ # Outils de génération de graphiques additionnels
    ├── solver/         # Cœur numérique (Solveurs implicites FDM d'ordre 1 et 2)
    └── verif/          # Outils de vérification (Génération de la MMS et calcul d'erreur L2)
