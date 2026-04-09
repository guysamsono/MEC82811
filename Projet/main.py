from src.plotter import temperature_plotter
from src.solver import solver_first_order, solver_second_order, compute_conservation_of_energy, mms_Temperature
from input import gen_input
from src.symmetry import test_symmetrie
from src.mms import generer_mms_simple
from src.error import *
from src.convergence import*

if __name__ == "__main__": 

    input_dict = gen_input()

    type_simul = 'temperature'        #type de simulation à réaliser : 'symmetry_test' ou 'temperature'
    order = '1'                         #ordre de la simulation : '1' pour ordre 1 et '2' pour ordre 2     

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
        
        #temperature_plotter(temperature, input_dict)

        f_T_MMS, f_source, f_bc_left, f_bc_right, f_bc_bottom, f_tinf_top = generer_mms_simple(input_dict, afficher_graphiques=True)

        temperature = solver_first_order(input_dict, sym_test = False, source_mms = f_source, 
                                         bc_left=f_bc_left, bc_right=f_bc_right, bc_bottom=f_bc_bottom, bc_top_tinf=f_tinf_top)
        temperature_plotter(temperature, input_dict)
        print(temperature)

    # Analyse de convergence avec MMS

    nx_list = [100,200,500,750]
    ny_list = [100,200,500,750]
    dx_list = [input_dict['b']/(nx-1) for nx in nx_list]
    dy_list = [input_dict['c']/(ny-1) for ny in ny_list]

    l1_list_x = []
    l2_list_x = []
    linf_list_x = []
    l1_list_y = []
    l2_list_y = []
    linf_list_y = []



    for nx in nx_list:
        input_dict['nx'] = nx
        input_dict['ny'] = 500

        f_T_MMS, f_source, f_bc_left, f_bc_right, f_bc_bottom, f_tinf_top = generer_mms_simple(input_dict, afficher_graphiques=False)

        temperature_sim = solver_first_order(input_dict, sym_test = False, source_mms = f_source, 
                                         bc_left=f_bc_left, bc_right=f_bc_right, bc_bottom=f_bc_bottom, bc_top_tinf=f_tinf_top)
        
        temperature_mms = mms_Temperature(input_dict, f_T_MMS)
        

        l1_list_x.append(norm_l1(temperature_sim, temperature_mms))
        l2_list_x.append(norm_l2(temperature_sim, temperature_mms))
        linf_list_x.append(norm_infinity(temperature_sim, temperature_mms))

    for ny in ny_list:
        input_dict['nx'] = 500
        input_dict['ny'] = ny

        f_T_MMS, f_source, f_bc_left, f_bc_right, f_bc_bottom, f_tinf_top = generer_mms_simple(input_dict, afficher_graphiques=False)

        temperature_sim = solver_first_order(input_dict, sym_test = False, source_mms = f_source, 
                                         bc_left=f_bc_left, bc_right=f_bc_right, bc_bottom=f_bc_bottom, bc_top_tinf=f_tinf_top)
        
        temperature_mms = mms_Temperature(input_dict, f_T_MMS)

        l1_list_y.append(norm_l1(temperature_sim, temperature_mms))
        l2_list_y.append(norm_l2(temperature_sim, temperature_mms))
        linf_list_y.append(norm_infinity(temperature_sim, temperature_mms))

    graph_error_log(input_dict,dx_list, l1_list_x, l2_list_x, linf_list_x,1,  
                    'x',
                    save_path="results/convergence_x.png",
                    show_fig=True,
                    xlabel=r"Taille de maille")
    
    graph_error_log(input_dict,dy_list, l1_list_y, l2_list_y, linf_list_y,1,  
                    'y',
                    save_path="results/convergence_y.png",
                    show_fig=True,
                    xlabel=r"Taille de maille")

