from src.plotter import temperature_plotter
from src.solver import solver_first_order, solver_second_order
from input import gen_input
from src.symmetry import test_symmetrie

if __name__ == "__main__": 

    input_dict = gen_input()

    type_simul = 'symmetry_test'        #type de simulation à réaliser : 'symmetry_test' ou 'temperature'
    order = '1'                         #ordre de la simulation : '1' pour ordre 1 et '2' pour ordre 2     

    if type_simul == 'symmetry_test':
        print('Test de symétrie en cours...')
        test_symmetrie(order, input_dict)
    
    if type_simul == 'temperature':
        print(f'Simulation de température en cours à ordre {order}...')
        if order == '1':
            temperature = solver_first_order(input_dict)
        else:
            temperature = solver_second_order(input_dict)
        
        temperature_plotter(temperature, input_dict)