"""
    Generador Congruencial Lineal
"""
LCG_A = 16_807
LCG_C = 0
LCG_M = (2 ** 31) - 1


"""
    Valor de d: Distintas dimensiones para realizar las estimaciones
""" 
TWO_DIMENSIONS  = 2
FIVE_DIMENSIONS = 5
TEN_DIMENSIONS  = 10

"""
    Tamaños de muestras
"""
SAMPLE_SIZE_SMALL  = 10_000
SAMPLE_SIZE_MEDIUM = 100_000
SAMPLE_SIZE_BIG    = 1_000_000

"""
    Calculo exacto de la integral
"""
INTEGRAL_VAL_D1 = 0.74682