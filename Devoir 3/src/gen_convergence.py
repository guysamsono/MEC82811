import numpy as np
import matplotlib.pyplot as plt
from lbm_accel import Generate_sample, LBM

def calcul_p_hat(k_list, ratio):
    p_hat = (np.log((k_list[0]-k_list[1])/(k_list[1]-k_list[2])))/(np.log(ratio))
    return p_hat

def gci_calculation(p_hat,p_f,k_list,ratio):
    rapport  = np.abs((p_hat-p_f)/p_f)
    if rapport <= 0.1:
        Fs = 1.25
        P = p_f
    elif rapport > 0.1:
        Fs = 3.0
        P = min(max(0.5,p_hat),p_f)

    GCI = Fs/(ratio**P -1)*np.abs(k_list[-2] - k_list[-1])

    return GCI
    

def gen_convergence_func(ratio, deltaP, NX, poro, mean_fiber_d, std_d, dx, filename, seed=101):
    
    nx_list = [NX/ratio, NX, NX*ratio]
    dx_list = [dx*ratio, dx, dx/ratio]
    k_list = []

    for i in range(len(nx_list)):
        print(f"Running LBM with NX={nx_list[i]} and dx={dx_list[i]}")
        d_equivalent = Generate_sample(seed, filename, mean_fiber_d, std_d, poro, int(nx_list[i]), dx_list[i])
        k_list.append(LBM(filename, int(nx_list[i]), deltaP, dx_list[i], d_equivalent))

    # Plotting the results
    plt.figure(figsize=(8, 6))
    plt.loglog(dx_list, k_list, marker='o', label='LBM Permeability')
    plt.xlabel('Grid Spacing (dx)')
    plt.ylabel('Permeability (k)')
    plt.title('Convergence of LBM Permeability with Grid Refinement')
    plt.grid(True, which="both", ls="--")
    try:
        plt.savefig('results/convergence_plot.png')
    except Exception as e:
        print(f"Error saving plot: {e}, does the 'results' directory exist?")
    plt.show()

    p_hat = calcul_p_hat(k_list, ratio)
    print(f"Estimated order of convergence (p_hat): {p_hat:.2f}")
    GCI = gci_calculation(p_hat, 2,k_list,ratio)
    print(f"Estimated Grid Convergence Index: {GCI:.2f}")