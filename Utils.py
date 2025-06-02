from typing import List
import numpy as np
from constants import INTEGRAL_VAL_D1

class Utils:
    """
    Clase con funciones utiles
    """
    
    @staticmethod
    def gaussian_function(x: float) -> float:
        """
        Función que se usa para estimar el valor de la integral con metódo
        de Monte Carlo.

        Args:
            x(float): valor con el que se inicializa la función gaussiana 
        
        Returns: 
            float: retorna el valor de la función gaussiana valuada en x
        """
        return np.e**(-x**2)
    
    @staticmethod
    def variance(estimations: List[float]) -> float:
        """
        Metódo para calcular la varianza entre estimaciones

        Args:
            estimations (List[float]): lista de estimaciones para
                                        calcular varianza

        Returns:
            float: Varianza de las estimaciones pasadas cómo parametro 
        """
        N = len(estimations)
        if N <= 1:
            raise Exception("Error: la varianza corregida requiere que N sea mayor a 1")
        
        return np.var(estimations, ddof=1)
    
    @staticmethod
    def cuadratic_error(estimations: List[float], d: int = 1) -> float:
        """
        Metódo para calcular el error cuadratico de una estimación
        restpecto al valor real de la integral de la función guasiana en
        un hipercubo.

        Args:
            estimations (List[float]): estimaciones que se utilizan para
                                       calcular el error cuadratico medio
                                       de las estimaciones para una dimension d.
            d (int): dimensión del hipercubo para calcular la integral

        Returns:
            float: Error cuadratico medio de la estimación respecto al resultado
                    exacto de la integral de la función gausiana en un hipercubo
                    de dim d
        """
        if d < 1:
            raise Exception("Error: la dimensión debe ser mayor a 1")
        
        integral_val = INTEGRAL_VAL_D1**d 
        cuadratic_error = 0
        for estimation in estimations:
            cuadratic_error += (estimation - integral_val)**2

        return cuadratic_error/len(estimations)