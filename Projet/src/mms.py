import os
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp



def generer_mms_simple(input_dict: dict, afficher_graphiques: bool = False, save_path: str = "results/MMS"):
    """
    Génère la solution manufacturée, le terme source et les graphiques associés.
    
    :param input_dict: Dictionnaire des paramètres physiques et d'entrée du problème
    :param afficher_graphique: Booléen pour bloquer/débloquer l'affichage des graphiques (désactiver pour le Bash)
    :return: f_C_MMS, f_source (fonctions lambdifiées utilisables par les solveurs)
    """

    # Extraction des paramètres physiques

    ny = input_dict['ny']
    nx = input_dict['nx']
    kappa = input_dict['k']
    rho = input_dict['rho']
    cp = input_dict['cp']
    b = input_dict['b']
    c = input_dict['c']
    d = input_dict['d']
    f = input_dict['f']
    h = input_dict['h']

    #Définition des variables symboliques
    x, y = sp.symbols('x y')

    A = 5

    # Solution manufacturée
    T_MMS = 100 + sp.sin(sp.pi*x/b) + sp.cos(sp.pi*y/(2*c)) + sp.sin(2*sp.pi*x/b)*sp.cos(2*sp.pi*y/(2*c))

    # Calcul des dérivées
    T_x = sp.diff(T_MMS, x)
    T_y = sp.diff(T_MMS, y)
    T_xx = sp.diff(T_x, x)
    T_yy = sp.diff(T_y, y)  

    # Calcul du terme source S(x,y)
    u_sym = (3*d)/(4*c) * (1 - (y/c)**2)

    source = rho*cp*u_sym*T_x - kappa*(T_xx + T_yy) - f

    # Conditions frontières
    # Gamma 2 : x = 0
    T_boundary_2 = sp.simplify(T_MMS.subs(x, 0))

    # Gamma 4 : x = b
    T_boundary_4 = sp.simplify(T_MMS.subs(x, b))

    # Gamma 3 : y = 0, Neumann = dT/dy(x,0)
    dT_dy_boundary_3 = sp.simplify(sp.diff(T_MMS, y).subs(y, 0))

    # Gamma 1 : y = c, Robin
    # -k dT/dy = h (T - T_inf)
    # donc T_inf(x) = T(x,c) + (k/h) dT/dy(x,c)
    T_top = sp.simplify(T_MMS.subs(y, c))
    dT_dy_top = sp.simplify(sp.diff(T_MMS, y).subs(y, c))
    T_inf_top = sp.simplify(T_top + (kappa / h) * dT_dy_top)


    # Affichage des dérivées
    print("Dérivée première en x:")
    print(T_x)
    print("Dérivée seconde en x:")
    print(T_xx)
    print("Dérivée première en y:")
    print(T_y)
    print("Dérivée seconde en y:")
    print(T_yy)
    print("Terme source :")
    print(source)
    print("\nCondition frontière gamma_2 :")
    print(T_boundary_2)
    print("\nCondition frontière gamma_3 :")
    print(dT_dy_boundary_3)
    print("\nCondition frontière gamma_4 :")
    print(T_boundary_4)

    # Conversion en fonctions Python
    f_T_MMS = sp.lambdify([x, y], T_MMS, "numpy")
    f_source = sp.lambdify([x, y], source, "numpy")

    f_T_MMS = sp.lambdify((x, y), T_MMS, "numpy")
    f_source = sp.lambdify((x, y), source, "numpy")
    f_bc_left = sp.lambdify(y, T_boundary_2, "numpy")
    f_bc_right = sp.lambdify(y, T_boundary_4, "numpy")
    f_bc_bottom = sp.lambdify(x, dT_dy_boundary_3, "numpy")
    f_tinf_top = sp.lambdify(x, T_inf_top, "numpy")

    # Création des maillages spatial pour les graphiques
    xdom = np.linspace(0, b, nx)
    ydom = np.linspace(0, c, ny)
    xi, yi = np.meshgrid(xdom, ydom, indexing='ij') 

    # Évaluation des fonctions sur le maillage pour les graphiques
    z_MMS = f_T_MMS(xi, yi)
    z_source = f_source(xi, yi)     

    # Graphiques
    if afficher_graphiques:
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(8, 4))
        contour1 = plt.contourf(xdom, ydom, z_MMS.T, 100, cmap='hot')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar(contour1, label='Temperature')
        plt.title('Temperature distribution for MMS solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(f"{save_path}/MMS_solution.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 4))
        contour2 = plt.contourf(xdom, ydom, z_source.T, 100, cmap='viridis')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar(contour2, label='Terme source S(x,y) [W/m³]')
        plt.title('Source distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(f"{save_path}/MMS_source.png", dpi=300)
        plt.close()

    return f_T_MMS, f_source, f_bc_left, f_bc_right, f_bc_bottom, f_tinf_top


