from rngs.RNG import RNG
from rngs import *
import numpy as np
from random import random
from scipy import stats

class Test:
    @staticmethod
    def KS_statistic(samples: np.ndarray) -> float:
        """
        Estadistico de Kolmogorov-Smirnov para una muestra

        Args:
            samples (np.ndarray): muestras que se reciben para el estadistico
        """
        n = len(samples)

        #Calculo la empírica
        Fe_samples = np.concatenate(([0], np.arange(1, n+1, 1) / n,[1]))

        # Distribución Uniforme Real
        F_values = stats.uniform.cdf(samples)

        # Calculamos el estadistico
        candidate_D1 = Fe_samples[1:n-1] - F_values[1:n-1]
        candidate_D2 = F_values[1:n-1] - F_values[0:n-2]

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
        # 
        d = Test.KS_statistic(x_samples)

        prob = 0
        for _ in range(Nsim):
            samples = set(random() for _ in range(Nsamples))
            #Ordeno las muestras
            u_samples = np.concatenate(([0], np.sort(list(samples)), [1]))
            d_sim = Test.KS_statistic(u_samples)
            if d_sim >= d:
                prob += 1
        return prob/Nsim >= 0.05
    

