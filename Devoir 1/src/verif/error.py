"""
Module de calcul des normes d'erreur (L1, L2, Infini).
"""
import numpy as np

def norm_l1(concentration, concentration_analytique):
    """
    Calcule la norme L1 (moyenne des erreurs absolues).

    :param concentration: Solution numérique
    :param concentration_analytique: Solution analytique
    :return: Norme L1
    """
    return np.mean(np.abs(concentration - concentration_analytique))


def norm_l2(concentration, concentration_analytique):
    """
    Calcule la norme L2 (erreur quadratique moyenne).

    :param concentration: Solution numérique
    :param concentration_analytique: Solution analytique
    :return: Norme L2
    """
    diff = concentration - concentration_analytique
    return np.sqrt(np.mean(diff**2))


def norm_infinity(concentration, concentration_analytique):
    """
    Calcule la norme infinie (erreur maximale absolue).

    :param concentration: Solution numérique
    :param concentration_analytique: Solution analytique
    :return: Norme Infinie
    """
    return np.max(np.abs(concentration - concentration_analytique))
