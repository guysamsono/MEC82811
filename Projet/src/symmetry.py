from src.solver import solver_first_order, solver_second_order
from src.plotter import temperature_plotter

def test_symmetrie(order, input_dict):

    assert order in ['1', '2'], "Order must be either 1 or 2"

    c = input_dict['c']
    input_dict['c'] = 2*c

    ny = input_dict['nx']
    input_dict['ny'] = 2*ny

    if order == '1':
        T_sym_first_order = solver_first_order(input_dict, True)
        temperature_plotter(T_sym_first_order, input_dict, 'symmetrised_domain_test_first_order.png')

    else:
        T_sym_second_order = solver_second_order(input_dict, True)
        temperature_plotter(T_sym_second_order, input_dict, 'symmetrised_domain_test_second_order.png')
    
    T_first_order = solver_first_order(input_dict, False)
    T_second_order = solver_second_order(input_dict, False)

    return 

    