def gen_input():
    
    input = {
        'a':0,              #borne inférieure du domaine (x)
        'b':10,             #borne supérieure du domaine (x)
        'c':4,              #demi-largeur du domaine (y)  
        'nx':10,            #nombre de points de discrétisation en x
        'ny':10,            #nombre de points de discrétisation en y
        'rho':1,            #densité
        'cp':1,             #chaleur spécifique
        'k':1,              #conductivité thermique
        'f':10,             #therme source
        'u':1,              #vitesse
        'temp_a':100,       #température à la borne a
        'temp_b':100,       #température à la borne b
        'q':10              #flux de chaleur sur les cotés du domaine
    }

    return input