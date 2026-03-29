def experimental(incert_1, incert_2, incert_3 = 0, incert_4 = 0, incert_5 = 0):
    '''
    Calcule l'incertitude totale à partir de plusieurs incertitudes individuelles expériemntale
    '''
    return (incert_1**2 + incert_2**2 + incert_3**2 + incert_4**2 + incert_5**2)**0.5