from src.plotter import temperature_plotter
from src.solver import *
from input import gen_input
from src.symmetry import test_symmetrie
from src.mms import generer_mms_simple, mms_convergence_analysis
from src.error import *
from src.convergence import*
from src.solution_verification import solution_verification, post_processing_verification

if __name__ == "__main__": 

    input_dict = gen_input()

    type_simul = 'temperature'   #type de simulation à réaliser : 'symmetry_test' ou 'temperature' ou 'temperature_mms ou solution_verification'
    order = '2'                            #ordre de la simulation : '1' pour ordre 1 et '2' pour ordre 2     

    if type_simul == 'symmetry_test':
        print('Test de symétrie en cours...')
        test_symmetrie(order, input_dict)
    
    if type_simul == 'temperature':
        print(f'Simulation de température en cours à ordre {order}...')
        if order == '1':
            temperature = solver_first_order(input_dict)
            srq = compute_conservation_of_energy(temperature, input_dict)
            print(f"Résidu de la conservation de l'énergie : {srq}")
        else:
            temperature = solver_second_order(input_dict)
            srq = compute_conservation_of_energy(temperature, input_dict)
            print(f"Résidu de la conservation de l'énergie : {srq}")
        
        temperature_plotter(temperature, input_dict)
        save_as_csv(temperature, input_dict, f"results/temperature_order_{order}.csv")
        save_input_as_csv(input_dict, f"results/input_parameters_order_{order}.csv")

    if type_simul == 'temperature_mms':

        mms_convergence_analysis(input_dict, order)

    if type_simul == 'solution_verification':

        solution_verification(input_dict,2)

    if type_simul == 'post_processing_verification':
        
        post_processing_verification(input_dict)

