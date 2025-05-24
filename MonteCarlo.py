from typing import Callable


class MonteCarlo:
    """
    Implementa el método de MonteCarlo multivariable
    """

    @staticmethod
    def method(Ndim: int, Nsamples: int,
               g: Callable[[float], float],
               random_method: Callable[[], float]) -> float:
        """
        Método de MonteCarlo para N variables

        Args:
            Ndim (int): Número de dimensiones
            Nsamples (int): Número de muestras
            g (Callable[[float], float]): Función a aplicar
            random_method (Callable[[], float]): Método generador de números random

        Returns:
            float: Estimación de la esperanza de g sobre un dominio uniforme.
        """
        integral = 0
        for _ in range(Nsamples):
            U_list = [random_method() for _ in range(Ndim)]
            integral += g(*U_list)
        return integral / Nsamples
