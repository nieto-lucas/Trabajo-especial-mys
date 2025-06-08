import numpy as np
from numpy.typing import ArrayLike
from typing import Callable

class TestHelpers:
    """
    Función auxiliares para tests, por ejemplo estadisticos.
    """

    @staticmethod
    def KS_statistic(samples: ArrayLike, G: Callable[[ArrayLike], ArrayLike]) -> float:
        """
        Estadistico de Kolmogorov-Smirnov para una muestra

        Args:
            samples (ArrayLike): muestras que se reciben para el estadistico
            G (Callable[[ArrayLike],ArrayLike]): función sobre la que se aplica
            el estadistico.
        
        Returns:
            (float): Resultado del estadistico.
        """
        n = len(samples)
        # Calculo la empírica
        Fe_samples = np.concatenate(([0], np.arange(1, n+1, 1) / n,[1]))
        # Distribución Uniforme Real
        G_values = G(samples)
        # Calculamos el estadistico
        candidate_D1 = Fe_samples[1:n-1] - G_values[1:n-1]
        candidate_D2 = G_values[1:n-1] - Fe_samples[0:n-2]
        # D = d
        candidates_D = np.concatenate((candidate_D1, candidate_D2))
        d = np.max(candidates_D)
        return d