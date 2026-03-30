from src.solver import solver_first_order, solver_second_order
from src.plotter import temperature_plotter
import numpy as np

def test_symmetrie(order, input_dict):

    assert order in ['1', '2'], "Order must be either 1 or 2"

    c0 = input_dict['c']
    ny0 = input_dict['ny']
    nx = input_dict['nx']

    input_dict['c'] = 2 * c0
    input_dict['ny'] = 2 * ny0

    if order == '1':
        t_sym = solver_first_order(input_dict, True)
        temperature_plotter(t_sym, input_dict, 'symmetrised_domain_test_first_order.png')
    else:
        t_sym = solver_second_order(input_dict, True)
        temperature_plotter(t_sym, input_dict, 'symmetrised_domain_test_second_order.png')

    # restore original domain
    input_dict['c'] = c0
    input_dict['ny'] = ny0

    if order == '1':
        t_half = solver_first_order(input_dict, False)
        upper_domain_temp = t_sym[:ny0 * nx]
        l2_norm = np.linalg.norm(upper_domain_temp - t_half)
        print(f'L2 norm between the two solutions: {l2_norm}')

    else:
        t_half = solver_second_order(input_dict, False)
        upper_domain_temp = t_sym[:ny0 * nx]
        l2_norm = np.linalg.norm(upper_domain_temp - t_half)
        print(f'L2 norm between the two solutions: {l2_norm}')

    return