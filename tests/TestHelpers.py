import numpy as np
from numpy.typing import ArrayLike
from typing import Callable

class TestHelpers:
    """
    Función auxiliares para tests, por ejemplo estadisticos.
    """

    @staticmethod
    def KS_statistic(Nsamples:int, samples: ArrayLike, G: Callable[[ArrayLike], ArrayLike]) -> float:
        """
        Estadistico de Kolmogorov-Smirnov para una determinada cantidad de muestras

        Args:
            samples (ArrayLike): muestras que se reciben para el estadistico
            G (Callable[[ArrayLike],ArrayLike]): función sobre la que se aplica
            el estadistico.
        
        Returns:
            (float): Resultado del estadistico.
        """


        # Calculo la empírica
        Fe_minus = np.arange(0, Nsamples) / Nsamples
        Fe_plus = np.arange(1, Nsamples + 1) / Nsamples

        # Distribución Uniforme Real y convierto en arreglo numpy
        G_values = np.asarray(G(samples))

        # Calculamos el estadistico
        candidate_D1 = Fe_plus - G_values
        candidate_D2 = G_values - Fe_minus

        # D = d
        candidates_D = np.concatenate((candidate_D1, candidate_D2))
        d = np.max(candidates_D)
        return d