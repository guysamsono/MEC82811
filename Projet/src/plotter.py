"""
Module de génération et sauvegarde de graphiques.

Fournit des fonctions pour tracer des champs de température en 2D,
des champs d'erreur, et des coupes 1D.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

def temperature_plotter(
        t_array, input_dict, title=None,
        filename='temperature_field.png', sym_test=False):
    """
    Affiche la distribution de température à partir d'un tableau 1D.

    :param t_array: tableau 1D de la température (taille nx*ny).
    :param input_dict: dictionnaire des paramètres ('nx', 'ny', 'k', 'b', 'c').
    :param title: titre du graphique (optionnel).
    :param filename: nom du fichier de sauvegarde.
    :param sym_test: booléen (affecte l'axe y pour le test de symétrie).
    :return: None (affiche le graphique et le sauvegarde).
    """
    b, c = input_dict['b'], input_dict['c']
    nx, ny = input_dict['nx'], input_dict['ny']

    y = np.linspace(-c, c, ny) if sym_test else np.linspace(0, c, ny)
    x = np.linspace(0, b, nx)

    t_mesh = t_array.reshape((ny, nx))

    plt.figure(figsize=(8, 4))
    plt.contourf(x, y, t_mesh, 100, cmap='hot')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(label='Temperature')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title if title else 'Temperature distribution')
    plt.tight_layout()

    save_full_path = os.path.join(input_dict['save_path'], filename)
    plt.savefig(save_full_path, dpi=300)
    plt.close()


def error_plotter(error, input_dict, filename='error_field.png'):
    b, c = input_dict['b'], input_dict['c']
    nx, ny = input_dict['nx'], input_dict['ny']

    x = np.linspace(0, b, nx)
    y = np.linspace(0, c, ny)

    error = error.reshape((ny, nx))
    error_log = np.log10(np.abs(error) + 1e-16)

    plt.figure(figsize=(8, 4))
    plt.contourf(x, y, error_log, 100, cmap='viridis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(label=r'$\log_{10}(|\mathrm{Error}|)$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Log error distribution')

    save_full_path = os.path.join(input_dict['save_path'], filename)
    plt.savefig(save_full_path, dpi=300)
    plt.close()


# pylint: disable=too-many-arguments, too-many-positional-arguments
def one_dimension_plotter(
        x, y, plot_dict, input_dict, last_graph=False,linestyle='-',
        color='blue', filename='one_dimension_plot_group.png'):
    """
    Affiche une courbe 1D.

    :param x: tableau 1D des abscisses.
    :param y: tableau 1D des ordonnées.
    :param plot_dict: dictionnaire de personnalisation ('xlabel', 'title'...).
    :param input_dict: dictionnaire contenant 'save_path'.
    :param last_graph: bool indiquant si c'est le dernier graphique à afficher.
    :param color: couleur de la courbe (par défaut 'blue').
    :param filename: nom du fichier de sauvegarde.
    :return: None.
    """
    plt.plot(x, y, label=f'{plot_dict["label"]}', color=color, linestyle=linestyle)
    plt.xlabel(plot_dict["xlabel"])
    plt.ylabel(plot_dict["ylabel"])
    plt.title(plot_dict["title"])
    plt.legend()
    plt.grid(True, alpha=0.3)

    if last_graph:
        plt.tight_layout()
        save_full_path = os.path.join(input_dict['save_path'], filename)
        plt.savefig(save_full_path, dpi=300)
        plt.close()
