import matplotlib.pyplot as plt
import numpy as np
import os

def temperature_plotter(T ,input_dict, filename='temperature_field.png'):

    a = input_dict['a']
    b = input_dict['b']
    c = input_dict['c']
    nx = input_dict['nx']
    ny = input_dict['ny']

    x = np.linspace(a, b, nx)
    y = np.linspace(a, c, ny)

    T = T.reshape((ny, nx))
    plt.contourf(x, y, T, 100, cmap='hot')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(label='Temperature')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Temperature distribution')
    
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

    a = input_dict['a']
    b = input_dict['b']
    c = input_dict['c']
    nx = input_dict['nx']
    ny = input_dict['ny']

    x = np.linspace(a, b, nx)
    y = np.linspace(a, c, ny)

    error = error.reshape((ny, nx))
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

