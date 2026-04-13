"""
Module de calcul des normes d'erreur (L1, L2, Infini).
"""
import numpy as np

def norm_l1(temperature, temperature_analytique):
    """
    Calcule la norme L1 (moyenne des erreurs absolues).

    :param temperature: Solution numérique
    :param temperature_analytique: Solution analytique
    :return: Norme L1
    """
    return np.mean(np.abs(temperature - temperature_analytique))


def norm_l2(temperature, temperature_analytique):
    """
    Calcule la norme L2 (erreur quadratique moyenne).

    :param temperature: Solution numérique
    :param temperature_analytique: Solution analytique
    :return: Norme L2
    """
    diff = temperature - temperature_analytique
    return np.sqrt(np.mean(diff**2))


def norm_infinity(temperature, temperature_analytique):
    """
    Calcule la norme infinie (erreur maximale absolue).

    :param temperature: Solution numérique
    :param temperature_analytique: Solution analytique
    :return: Norme Infinie
    """
    return np.max(np.abs(temperature - temperature_analytique))
