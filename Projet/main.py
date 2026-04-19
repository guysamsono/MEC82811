"""
Module principal de simulation de transport de chaleur (MEC8211).

Ce script orchestre la résolution de l'équation de convection-diffusion stationnaire
sur un domaine rectangulaire. Il permet de piloter l'ensemble du processus de
Vérification et Validation (V&V) à travers plusieurs modes d'exécution.
"""
from input import gen_input
from src.plotter import temperature_plotter
from src.solver import (
    solver_first_order, solver_second_order,
    compute_conservation_of_energy,
    save_as_csv, save_input_as_csv)
from src.symmetry import test_symmetrie
from src.mms import mms_convergence_analysis
from src.solution_verification import solution_verification, post_processing_verification

if __name__ == "__main__":
    input_dict = gen_input()

    SIMUL_TYPE = 'full_simulation'         #type de simulation à réaliser :
                                           # 'symmetry_test' ou 'temperature' ou 'temperature_mms'
                                           # ou 'solution_verification' ou 'full_simulation'
    ORDER = '2'                            #ordre de la simulation : '1' ou '2'

    if SIMUL_TYPE == 'symmetry_test':
        print('Test de symétrie en cours...')
        test_symmetrie(ORDER, input_dict, scheme='upwind')

    if SIMUL_TYPE == 'temperature':
        print(f'Simulation de température en cours à ordre {ORDER}...')
        if ORDER == '1':
            temperature = solver_first_order(input_dict)
            srq = compute_conservation_of_energy(temperature, input_dict)
            print(f"Résidu de la conservation de l'énergie : {srq}")
        else:
            temperature = solver_second_order(input_dict,'upwind')
            srq = compute_conservation_of_energy(temperature, input_dict)
            print(f"Résidu de la conservation de l'énergie : {srq}")

        temperature_plotter(temperature, input_dict)
        save_as_csv(temperature, input_dict,
                    f"{input_dict['save_path']}/temperature_order_{ORDER}.csv")
        save_input_as_csv(input_dict,
                          f"{input_dict['save_path']}/input_parameters_order_{ORDER}.csv")

    if SIMUL_TYPE == 'temperature_mms':
        print('Vérification du code (MMS) en cours...')
        mms_convergence_analysis(input_dict, ORDER, scheme='central')

    if SIMUL_TYPE == 'solution_verification':
        print('Vérification de la solution en cours...')
        solution_verification(input_dict,ORDER, scheme='upwind')

    if SIMUL_TYPE == 'post_processing_verification':
        post_processing_verification(input_dict)

    if SIMUL_TYPE == 'full_simulation':
        print('\n' + '='*60)
        print(f'LANCEMENT DE LA SIMULATION COMPLÈTE (Ordre {ORDER})')
        print('='*60 + '\n')

        print('--- ÉTAPE 1 : Test de symétrie ---')
        test_symmetrie(ORDER, input_dict, scheme='upwind')
        print('Test de symétrie terminé.\n')

        print('--- ÉTAPE 2 : Calcul de la température et conservation de l\'énergie---')
        if ORDER == '1':
            temperature = solver_first_order(input_dict)
        else:
            temperature = solver_second_order(input_dict, scheme='upwind')

        srq = compute_conservation_of_energy(temperature, input_dict)
        print(f"Résidu de la conservation de l'énergie : {srq}")

        temperature_plotter(temperature, input_dict)
        save_as_csv(temperature, input_dict,
                    f"{input_dict['save_path']}/temperature_order_{ORDER}.csv")
        save_input_as_csv(input_dict,
                          f"{input_dict['save_path']}/input_parameters_order_{ORDER}.csv")
        print('Calcul de la température terminé.\n')

        print('--- ÉTAPE 3 : Vérification de code (MMS) ---')
        mms_convergence_analysis(input_dict, ORDER, scheme='central')
        print('Vérification de code terminée.\n')

        print('--- ÉTAPE 4 : Vérification de solution ---')
        solution_verification(input_dict, int(ORDER), scheme='upwind')
        print('Vérification de solution terminée.\n')

        print('--- ÉTAPE 5 : Post-processing final ---')
        post_processing_verification(input_dict)
        print('Post-processing terminé.\n')

        print('='*60)
        print('SIMULATION COMPLÈTE TERMINÉE AVEC SUCCÈS')
        print('='*60 + '\n')
