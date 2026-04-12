"""
Module dédié aux tests de symétrie.

Ce module vérifie que la solution calculée sur la moitié supérieure du domaine
(avec une condition de Neumann) correspond parfaitement à la solution
calculée sur le domaine complet.
"""
import os
import numpy as np
from src.solver import solver_first_order, solver_second_order
from src.plotter import temperature_plotter, error_plotter, one_dimension_plotter


# pylint: disable=too-many-locals
def test_symmetrie(order, input_dict, scheme):

    '''
    Test de symétrie pour les schémas d'ordre 1 et 2.
    param order: ordre du schéma à tester ('1' ou '2')
    param input_dict: dictionnaire contenant les paramètres du problème
    return: None
    '''

    sym_dir = os.path.join(input_dict['save_path'], 'SYMMETRY')
    os.makedirs(sym_dir, exist_ok=True)

    assert order in ['1', '2'], "Order must be either 1 or 2"

    if order == '1':
        t_normal_order = solver_first_order(input_dict)
        temperature_plotter(
            t_normal_order,
            input_dict,
            title='Champ de température - domaine normal - ordre 1',
            filename='SYMMETRY/normal_domain_test_first_order.png'
        )
    else:
        t_normal_order = solver_second_order(input_dict, scheme)
        temperature_plotter(
            t_normal_order,
            input_dict,
            title=f'Champ de température - domaine normal - ordre 2 ({scheme})',
            filename='SYMMETRY/normal_domain_test_second_order.png'
        )

    sym_input_dict = input_dict.copy()
    sym_input_dict['ny'] = 2 * input_dict['ny']

    if order == '1':
        t_sym = solver_first_order(sym_input_dict, True)
        temperature_plotter(
            t_sym,
            sym_input_dict,
            title='Champ de température - domaine symétrisé - ordre 1',
            filename='SYMMETRY/symmetrised_domain_test_first_order.png',
            sym_test=True
        )
    else:
        t_sym = solver_second_order(sym_input_dict, scheme, True)
        temperature_plotter(
            t_sym,
            sym_input_dict,
            title=f'Champ de température - domaine symétrisé - ordre 2 ({scheme})',
            filename='SYMMETRY/symmetrised_domain_test_second_order.png',
            sym_test=True
        )

    nx = input_dict['nx']
    ny = input_dict['ny']

    error = t_normal_order - t_sym[ny * nx : 2 * ny * nx]
    l2_norm = np.linalg.norm(error)
    print(f"norme L2 de l'erreur sur le domaine symétrisé: {l2_norm}")

    error_plotter(error, input_dict, 'SYMMETRY/symetry_error_field.png')

    plot_dict = {
        'xlabel': 'y',
        'ylabel': 'Temperature',
        'title': 'Temperature distribution along y-axis for symmetry test at middle of the domain',
        'label': 'Temperature on half domain'
    }

    t_normal_order_reshaped = t_normal_order.reshape((ny, nx))
    t_sym_reshaped = t_sym.reshape((sym_input_dict['ny'], nx))
    y = np.linspace(0, input_dict['c'], ny)
    j_mid = nx // 2

    one_dimension_plotter(
        y,
        t_normal_order_reshaped[:, j_mid],
        plot_dict,
        input_dict,
        last_graph=False,
        color='blue',
        filename='SYMMETRY/symmetry_1d.png',
    )

    plot_dict['label'] = 'Temperature on symmetrised domain'
    one_dimension_plotter(
        y,
        np.flip(t_sym_reshaped[:ny, j_mid]),
        plot_dict,
        input_dict,
        last_graph=True,
        color='red',
        filename='SYMMETRY/symmetry_1d.png'
    )

    y2 = np.linspace(-input_dict['c'], input_dict['c'], sym_input_dict['ny'])
    one_dimension_plotter(
        y2,
        t_sym_reshaped[:, j_mid],
        plot_dict,
        input_dict,
        last_graph=True,
        color='green',
        filename='SYMMETRY/symmetry_1d_full.png'
    )
