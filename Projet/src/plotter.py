import matplotlib.pyplot as plt
import numpy as np
import os

def temperature_plotter(T ,input_dict, title =None, filename='temperature_field.png', sym_test=False):

    '''
    Affiche la distribution de température à partir d'un tableau 1D de températures.
    param T: tableau 1D de la température à chaque point du maillage (taille nx*ny)
    param input_dict: dictionnaire contenant les paramètres du problème (doit inclure 'nx', 'ny', 'k', 'b', 'c')
    param title: titre du graphique (optionnel)
    param filename: nom du fichier de sauvegarde (optionnel, par défaut 'temperature_field.png')
    param sym_test: booléen indiquant si le test de symétrie est en cours (affecte l'axe y)
    return: None (affiche le graphique et le sauvegarde)'''


    b = input_dict['b']
    c = input_dict['c']
    nx = input_dict['nx']
    ny = input_dict['ny']

    if sym_test:
        y = np.linspace(-c, c, ny)
    else:
        y = np.linspace(0, c, ny)

    x = np.linspace(0, b, nx)

    T = T.reshape((ny, nx))

    plt.figure(figsize=(8, 4))
    plt.contourf(x, y, T, 100, cmap='hot')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(label='Temperature')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title if title else 'Temperature distribution')
    
    file_path = os.path.join('results', filename)

    try:
        plt.savefig(file_path)

    except Exception as err:
        print(f'{err},le classeur results/sera crée')
        os.mkdir('results/')
        plt.savefig(file_path)

    plt.close()
    return

def error_plotter(error, input_dict, filename='error_field.png'):

    b = input_dict['b']
    c = input_dict['c']
    nx = input_dict['nx']
    ny = input_dict['ny']

    x = np.linspace(0, b, nx)
    y = np.linspace(0, c, ny)

    error = error.reshape((ny, nx))

    plt.figure(figsize=(8, 4))
    plt.contourf(x, y, error, 100, cmap='viridis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(label='Error')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Error distribution')
    
    file_path = os.path.join('results', filename)

    try:
        plt.savefig(file_path)

    except Exception as err:
        print(f'{err},le classeur results/sera crée')
        os.mkdir('results/')
        plt.savefig(file_path)

    plt.close()
    return

def one_dimension_plotter(x, y, plot_dict, last_graph=False, color='blue', filename='one_dimension_plot_group.png'):

    '''
    Affiche une courbe 1D.
    param x: tableau 1D des abscisses
    param y: tableau 1D des ordonnées
    param plot_dict: dictionnaire contenant les éléments de personnalisation du graphique (doit inclure 'xlabel', 'ylabel', 'title', 'label')
    param last_graph: booléen indiquant si c'est le dernier graphique à afficher (affecte la grille et la fermeture du graphique)
    param color: couleur de la courbe (optionnel, par défaut 'blue')
    param filename: nom du fichier de sauvegarde (optionnel, par défaut 'one_dimension_plot_group.png')
    return: None (affiche le graphique et le sauvegarde)'''

    plt.plot(x, y, label=f'{plot_dict["label"]}', color=color)
    plt.xlabel(plot_dict["xlabel"])
    plt.ylabel(plot_dict["ylabel"])
    plt.title(plot_dict["title"])
    plt.legend()

    file_path = os.path.join('results', filename)

    try:
        plt.savefig(file_path)

    except Exception as err:
        print(f'{err},le classeur results/sera crée')
        os.mkdir('results/')
        plt.savefig(file_path)

    if last_graph:
        plt.grid()
        plt.close()

    return

