from typing import Callable, Tuple, List
from rngs.RNG import RNG
from time import time
import numpy as np

class MonteCarlo:
    """
    Implementa el método de MonteCarlo
    """

    @staticmethod
    def method(Nsamples: int,
               g: Callable[[float], float],
               rng: RNG) -> float:
        """
        Método de MonteCarlo para 1 variable

        Args:
            Nsamples (int): Número de muestras
            g (Callable[[float], float]): Función a aplicar
            rng (RNG): objeto de la clase RNG

        Returns:
            float: Estimación de la esperanza de g sobre un dominio uniforme.
        """
        integral = 0
        for _ in range(Nsamples):
            U = rng.rand01()
            integral += g(U)
        return integral / Nsamples
    
    @staticmethod
    def get_parcials_method_Nvars(Nsamples: int,
                                g: Callable[[np.ndarray], float],
                                rng: RNG,
                                Nvars: int) -> List[Tuple[List[float], float]]:
        """
        Método de MonteCarlo para Nvars-variable

        Args:
            Nsamples (int): Número de muestras
            g (Callable[[float], float]): Función a aplicar
            rng: (RNG): objeto de la clase RNG
            Nvars (int): 

        Returns:
            List[Tuple[List[float],float]]: Lista con las uniformes generadas
            por iteración de Monte Carlo y el resultado de valuar la función en
            en esas uniformes generadas.
        """
        parcial_estims = []
        integral = 0
        for _ in range(Nsamples):
            uniforms = np.array([rng.rand01() for _ in range(Nvars)])
            parcial_result = g(uniforms)
            integral += parcial_result
            parcial_estims.append((uniforms, parcial_result))
        return parcial_estims
