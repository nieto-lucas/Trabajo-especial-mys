from rngs.RNG import RNG
from rngs import *
import numpy as np
from numpy.typing import ArrayLike
from typing import Callable
from random import random
from scipy import stats

class Test:
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

    @staticmethod
    def test_Kolmogorov_Smirnov(rng:RNG, Nsamples:int, Nsim: int) -> bool:
        """
        Test de Kolmogorov_Smirnov con H0: "las muestras generadas por rng
        son uniformes en [0, 1]" y confianza de 95%.

        Args:
            Nsamples (int): numero de muestras por iteracion
            rng (RNG): objeto de la clase RNG para obtener muestras
            Nsim (int): numero de simulaciones para estimar el p-valor
        
        Returns:
            (bool): Resultado de si rechaza o no la H0
        """
        # Set para no generar muestras repetidas
        samples = set(rng.rand01() for _ in range(Nsamples))
        # Ordeno las muestras
        x_samples = np.concatenate(([0], np.sort(list(samples)), [1]))
        # Utiliza como función F la func de distrib acumulada de la unif.
        d = Test.KS_statistic(samples=x_samples, G=stats.uniform.cdf)

        prob = 0
        for _ in range(Nsim):
            samples = set(random() for _ in range(Nsamples))
            # Ordeno las muestras
            u_samples = np.concatenate(([0], np.sort(list(samples)), [1]))
            # Función identidad: G(u) = u
            d_sim = Test.KS_statistic(samples=u_samples, G=lambda x: x)
            if d_sim >= d:
                prob += 1
        return prob/Nsim >= 0.05