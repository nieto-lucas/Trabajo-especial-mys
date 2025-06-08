from rngs.RNG import RNG
from rngs import *
import numpy as np
from random import random
from scipy import stats
from tests.TestHelpers import TestHelpers

class Test:
    """
    Tests para verificar la corrección de rngs implementados.
    """

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
        d = TestHelpers.KS_statistic(samples=x_samples, G=stats.uniform.cdf)

        prob = 0
        for _ in range(Nsim):
            samples = set(random() for _ in range(Nsamples))
            # Ordeno las muestras
            u_samples = np.concatenate(([0], np.sort(list(samples)), [1]))
            # Función identidad: G(u) = u
            d_sim = TestHelpers.KS_statistic(samples=u_samples, G=lambda x: x)
            if d_sim >= d:
                prob += 1
        return prob/Nsim >= 0.05