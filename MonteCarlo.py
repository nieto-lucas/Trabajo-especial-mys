from typing import Callable, Tuple
from rngs.RNG import RNG
from time import time

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
            random_method (rng: RNG): objeto de la clase RNG

        Returns:
            float: Estimación de la esperanza de g sobre un dominio uniforme.
        """
        integral = 0
        for _ in range(Nsamples):
            U = rng.rand01()
            integral += g(U)
        return integral / Nsamples
    
    @staticmethod
    def time_method(Nsamples: int,
                    g: Callable[[float], float],
                    rng: RNG) -> Tuple[float, float]:
        """
        Método de MonteCarlo para 1 variable

        Args:
            Nsamples (int): Número de muestras
            g (Callable[[float], float]): Función a aplicar
            random_method (rng: RNG): objeto de la clase RNG

        Returns:
            Tuple[float, float]: 
                0. Tiempo que demora el metodo de Monte Carlo en segundos.
                1. Estimación de la esperanza de g sobre un dominio uniforme.
        """
        start = time()
        estimation = MonteCarlo.method(Nsamples, g, rng)
        end = time()
        time_elapsed = end - start
        return time_elapsed, estimation