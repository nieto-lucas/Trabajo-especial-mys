from typing import Callable
from rngs.RNG import RNG

class MonteCarlo:
    """
    Implementa el método de MonteCarlo multivariable
    """

    @staticmethod
    def method(Ndim: int, Nsamples: int,
               g: Callable[[float], float],
               rng: RNG) -> float:
        """
        Método de MonteCarlo para N variables

        Args:
            Ndim (int): Número de dimensiones
            Nsamples (int): Número de muestras
            g (Callable[[float], float]): Función a aplicar
            random_method (rng: RNG): objeto de la clase RNG

        Returns:
            float: Estimación de la esperanza de g sobre un dominio uniforme.
        """
        integral = 0
        for _ in range(Nsamples):
            U_list = [rng.rand01() for _ in range(Ndim)]
            integral += g(*U_list)
        return integral / Nsamples
