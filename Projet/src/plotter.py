import matplotlib.pyplot as plt
import numpy as np
import os

def temperature_plotter(T ,input_dict):

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
    
    try:
        plt.savefig('results/temperature_field.png')

    except Exception as err:
        print(f'{err},le classeur results/sera crée')
        os.mkdir('results/')
        plt.savefig('results/temperature_field.png')


