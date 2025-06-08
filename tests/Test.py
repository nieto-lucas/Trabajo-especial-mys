from rngs.RNG import RNG
from rngs import *
import numpy as np
from random import random
from scipy import stats
from tests.TestHelpers import TestHelpers
from visuals.Printers import Printers

class Test:
    """
    Tests para verificar la corrección de rngs implementados.
    """

    @staticmethod
    def test_Kolmogorov_Smirnov(rng:RNG, Nsamples:int, Nsim: int):
        """
        Test de Kolmogorov_Smirnov con H0: "las muestras generadas por rng
        son uniformes en [0, 1]" y confianza de 95%.

        Args:
            Nsamples (int): numero de muestras por iteracion
            rng (RNG): objeto de la clase RNG para obtener muestras
            Nsim (int): numero de simulaciones para estimar el p-valor
        """
        # Genero muestras del generador rng
        samples = np.asarray([rng.rand01() for _ in range(Nsamples)])

        # Ordeno las muestras
        x_samples = np.sort(list(samples))

        Nsamples = len(x_samples)

        # Utiliza como función F la func de distrib acumulada de la unif.
        d = TestHelpers.KS_statistic(Nsamples=Nsamples ,samples=x_samples, G=stats.uniform.cdf)

        value_p = 0
        for _ in range(Nsim):
            samples = set(random() for _ in range(Nsamples))
            # Ordeno las muestras
            u_samples = np.sort(list(samples))  
            #Extraigo el tamaño
            u_Nsamples = len(u_samples) 
            # Función identidad: G(u) = u
            d_sim = TestHelpers.KS_statistic(Nsamples=u_Nsamples, samples=u_samples, G=lambda x: x)

            if d_sim >= d:
                value_p += 1
        
        value_p = value_p/Nsim

        Printers.print_testKS_results(rng=rng.name(), test_results=(d, value_p), alpha=0.05)
