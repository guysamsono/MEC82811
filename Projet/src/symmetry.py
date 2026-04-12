from src.solver import solver_first_order, solver_second_order
from src.plotter import temperature_plotter, error_plotter, one_dimension_plotter
import numpy as np

def test_symmetrie(order, input_dict):

    '''
    Test de symétrie pour les schémas d'ordre 1 et 2.
    param order: ordre du schéma à tester ('1' ou '2')
    param input_dict: dictionnaire contenant les paramètres du problème (doit inclure 'nx', 'ny', 'k', 'b', 'c', 'rho', 'cp', 'f', 'h')
    return: None (affiche les graphiques et imprime la norme de l'erreur)
    '''

    assert order in ['1', '2'], "Order must be either 1 or 2"

    if order == '1':
        t_normal_order = solver_first_order(input_dict)
        temperature_plotter(t_normal_order, input_dict, 'normal_domain_test_first_order.png')
    else:
        t_normal_order = solver_second_order(input_dict)
        temperature_plotter(t_normal_order, input_dict, 'normal_domain_test_second_order.png')

    ny0 = input_dict['ny']
    input_dict['ny'] = 2 * ny0

    if order == '1':
        t_sym = solver_first_order(input_dict, True)
        temperature_plotter(t_sym, input_dict, 'symmetrised_domain_test_first_order.png', True)
    else:
        t_sym = solver_second_order(input_dict, True)
        temperature_plotter(t_sym, input_dict, 'symmetrised_domain_test_second_order.png', True)

    error = t_normal_order - t_sym[ny0*input_dict['nx']:ny0**2*input_dict['nx']]
    l2_norm = np.linalg.norm(error)
    print(f"norme L2 de l'erreur sur le domaine symétrisé: {l2_norm}")

    input_dict['ny'] = ny0

    error_plotter(error, input_dict, 'symetry_error_field.png')

    plot_dict = {
    'xlabel': 'y',
    'ylabel': 'Temperature',
    'title': 'Temperature distribution along y-axis for symmetry test at middle of the domain',
    'label': 'Temperature on half domain'
    }

    nx = input_dict['nx']
    ny = input_dict['ny']

    t_normal_order_reshaped = t_normal_order.reshape((ny, nx))
    t_sym_reshaped = t_sym.reshape((2 * ny0, nx))
    y = np.linspace(0, input_dict['c'], ny)
    j_mid = nx // 2

    one_dimension_plotter(y, t_normal_order_reshaped[:, j_mid], plot_dict, last_graph=False, color='blue', filename='symmetry_1d.png')
    plot_dict['label'] = 'Temperature on symmetrised domain'
    one_dimension_plotter(y, np.flip(t_sym_reshaped[:ny0, j_mid]), plot_dict, last_graph=True, color='red', filename='symmetry_1d.png')
    
    y2 = np.linspace(-input_dict['c'], input_dict['c'], 2*ny)
    one_dimension_plotter(y2, t_sym_reshaped[:, j_mid], plot_dict, last_graph=True, color='green', filename='symmetry_1d_full.png')
    
    return