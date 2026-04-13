import datetime
import os

def gen_input():

    '''
    Génère un dictionnaire d'entrée pour les fonctions de résolution et de test.
    return: input_dict (dictionnaire contenant les paramètres du problème)
    '''
    input_dict = {
    'b': 0.1,
    'c': 0.01,
    'd': 8.0e-5,
    'nx': 101,
    'ny': 101,
    'rho': 998,
    'cp': 4182,
    'k': 0.60,
    'f': 0.0,
    'temp_a': 313.15,
    'temp_b': 305.15,
    'h': 300,
    'tinf': 293.15
}

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(BASE_DIR, "results", f"run_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    input_dict['save_path'] = session_dir

    return input_dict
