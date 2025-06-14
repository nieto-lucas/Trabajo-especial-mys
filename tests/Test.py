import numpy as np
from numpy.typing import ArrayLike
from random import random
from scipy.stats import kstest, uniform
from tests.TestHelpers import TestHelpers
from visuals.Printers import Printers


class Test:
    """
    Tests para verificar la corrección de rngs implementados.
    """

    @staticmethod
    def test_Kolmogorov_Smirnov(rng_name:str, samples: ArrayLike, Nsim: int):
        """
        Test de Kolmogorov_Smirnov con H0: "las muestras generadas por rng
        son uniformes en [0, 1]" y confianza de 95%.

        Args:
            rng (RNG): objeto de la clase RNG para obtener muestras
            Nsim (int): numero de simulaciones para estimar el p-valor
        """
        # Ordeno las muestras
        samples = np.sort(samples)

        # Cantidad de muestras
        Nsamples = len(samples)

        # Utiliza como función F la func de distrib acumulada de la unif.
        d = TestHelpers.KS_statistic(
                Nsamples=Nsamples,
                samples=samples,
                G=uniform.cdf)

        #Estimación del p_valor
        value_p = 0
        for _ in range(Nsim):
            # Genero muestras y las ordeno
            samples = np.sort([random() for _ in range(Nsamples)])

            # Función identidad: G(u) = u
            d_sim = TestHelpers.KS_statistic(
                Nsamples=Nsamples, samples=samples, G=lambda x: x)

            if d_sim >= d:
                value_p += 1

        value_p = value_p/Nsim

        Printers.print_testKS_results(
            rng=rng_name,
            test_results=(d, value_p),
            alpha=0.05)

    def test_KS_scipy(rng_name:str, samples:ArrayLike):
        """
        Test de Kolmogorov_Smirnov con H0: "las muestras generadas por rng
        son uniformes en [0, 1]" y confianza de 95%. Utilizando scipy

        Args:
            rng_name (str): Nombre del generador
            samples (ArrayLike): Muestras
        """
        scipy_results = kstest(samples, cdf="uniform")[:2]
        D = scipy_results[0]
        value_p = scipy_results[1]
        Printers.print_testKS_scipy(rng=rng_name, test_results=(D, value_p), alpha=0.05)