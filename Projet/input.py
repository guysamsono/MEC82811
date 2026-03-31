def gen_input():
    
    input = {
        'a':0,              #borne inférieure du domaine (x)
        'b':20,             #borne supérieure du domaine (x)
        'c':4,              #demi-largeur du domaine (y)  
        'nx':100,            #nombre de points de discrétisation en x
        'ny':20,            #nombre de points de discrétisation en y
        'rho':1,            #densité
        'cp':1,             #chaleur spécifique
        'k':1,              #conductivité thermique
        'f':10,             #therme source
        'u':1,              #vitesse
        'temp_a':100,       #température à la borne a
        'temp_b':100,       #température à la borne b
        'h':5,              #coefficient de transfert thermique sur les cotés du domaine
        'tinf':40          #température de référence pour les conditions de robin
    }

    return input