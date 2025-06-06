from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo import MonteCarlo
from constants import INTEGRAL_VAL_D1
from rngs.RNG import RNG
from rngs.Xorshift32 import Xorshift
from rngs.MersenneTwister import MersenneTwister
from rngs.LCG import LCG

class Utils:
    """
    Clase con funciones utiles
    """
    
    @staticmethod
    def gaussian_function(x: float) -> float:
        """
        Función que se usa para estimar el valor de la integral con metódo
        de Monte Carlo.

        Args:
            x(float): valor con el que se inicializa la función gaussiana 
        
        Returns: 
            float: retorna el valor de la función gaussiana valuada en x
        """
        return np.e**(-x**2)
    
    @staticmethod
    def variance(estimations: List[float]) -> float:
        """
        Metódo para calcular la varianza entre estimaciones

        Args:
            estimations (List[float]): lista de estimaciones para
                                        calcular varianza

        Returns:
            float: Varianza de las estimaciones pasadas cómo parametro 
        """
        N = len(estimations)
        if N <= 1:
            raise Exception("Error: la varianza corregida requiere que N sea mayor a 1")
        
        return np.var(estimations, ddof=1)
    
    @staticmethod
    def cuadratic_error(estimations: List[float], d: int = 1) -> float:
        """
        Metódo para calcular el error cuadratico de una estimación
        restpecto al valor real de la integral de la función guasiana en
        un hipercubo.

        Args:
            estimations (List[float]): estimaciones que se utilizan para
                                       calcular el error cuadratico medio
                                       de las estimaciones para una dimension d.
            d (int): dimensión del hipercubo para calcular la integral

        Returns:
            float: Error cuadratico medio de la estimación respecto al resultado
                    exacto de la integral de la función gausiana en un hipercubo
                    de dim d
        """
        if d < 1:
            raise Exception("Error: la dimensión debe ser mayor a 1")
        
        integral_val = INTEGRAL_VAL_D1**d 
        cuadratic_error = 0
        for estimation in estimations:
            cuadratic_error += (estimation - integral_val)**2

        return cuadratic_error/len(estimations)
    
    @staticmethod
    def get_samples(Nsamples: int, Nsim: int, seed: int, d: int = 1) -> Dict[str, List[float]]:
        """
        Metódo para obtener muestras de estimaciones de Monte Carlo con parametros
        especificados a tráves de los argumentos y los tres rngs, MersenneTwister,
        LCG y Xorshift

        Args:
            Nsamples (int): numero de muestras que se desean obtener
            Nsim (int): numero de simulaciones de Monte Carlo por muestra
            d (int): dimension del hipercubo para calcular la integral 
        
        Returns:
            Dict[str, List[float]]: donde las claves se corresponden a los
                                    nombres de las clases de rngs: LCG, 
                                    Xorshift y MersenneTwister 
        """
        if d < 1:
            raise Exception("Error: la dimensión debe ser mayor a 1")

        # inicialización de los rngs
        lcg, lcg_estimation_samples = LCG(seed), []
        xorshift, xorshift_estimation_samples = Xorshift(seed), []
        mt, mt_estimation_samples = MersenneTwister(seed), []
        # cada rng recolecta muestras
        for _ in range(Nsamples): 
            lcg_estimation_samples.append(
                MonteCarlo.method(
                    Nsamples=Nsim, 
                    g=Utils.gaussian_function,
                    rng=lcg
                ) ** d
            )
            xorshift_estimation_samples.append(
                MonteCarlo.method(
                    Nsamples=Nsim, 
                    g=Utils.gaussian_function,
                    rng=xorshift
                ) ** d
            )
            mt_estimation_samples.append(
                MonteCarlo.method(
                    Nsamples=Nsim, 
                    g=Utils.gaussian_function,
                    rng=mt
                ) ** d
            )
        dict_samples = {
            "LCG" : lcg_estimation_samples,
            "Xorshift" : xorshift_estimation_samples,
            "MersenneTwister": mt_estimation_samples
        }
        return dict_samples

    @staticmethod
    def compare_var(Nsamples: int, Nsim: int, d: int = 1) -> Dict[str, float]:
        """
        Metódo para comparar varianza entre muestras de estimaciones de Monte Carlo
        para todos los rngs: LCG, Xorshift, MersenneTwister

        Args:
            Nsamples (int): numero de muestras que se desean obtener
            Nsim (int): numero de simulaciones de Monte Carlo por muestra
            d (int): dimension del hipercubo para calcular la integral 
        
        Returns:
            Dict[str, float]: varianzas, donde las claves se corresponden a 
                              los nombres de las clases de rngs: LCG, 
                              Xorshift y MersenneTwister 
        """
        try:
            samples = Utils.get_samples(Nsamples=Nsamples,
                                        Nsim=Nsim,
                                        seed=1234567,
                                        d=d)
            samples["LCG"] = Utils.variance(samples["LCG"])
            samples["Xorshift"] = Utils.variance(samples["Xorshift"])
            samples["MersenneTwister"] = Utils.variance(samples["MersenneTwister"])
            return samples
        except Exception as e:
            raise e
    
    @staticmethod
    def compare_cuadratic_error(Nsamples: int, Nsim: int, d: int = 1):
        """
        Metódo para comparar varianza entre muestras de estimaciones de Monte Carlo
        para todos los rngs: LCG, Xorshift, MersenneTwister

        Args:
            Nsamples (int): numero de muestras que se desean obtener
            Nsim (int): numero de simulaciones de Monte Carlo por muestra
            d (int): dimension del hipercubo para calcular la integral 
        
        Returns:
            Dict[str, float]: errores cuadraticos medios, donde las claves 
                              se corresponden a los nombres de las clases de 
                              rngs: LCG, Xorshift y MersenneTwister 
        """
        try:
            samples = Utils.get_samples(Nsamples=Nsamples,
                                        Nsim=Nsim,
                                        seed=1234567,
                                        d=d)
            samples["LCG"] = Utils.cuadratic_error(samples["LCG"])
            samples["Xorshift"] = Utils.cuadratic_error(samples["Xorshift"])
            samples["MersenneTwister"] = Utils.cuadratic_error(samples["MersenneTwister"])
            return samples
        except Exception as e:
            raise e

    @staticmethod    
    def plot_3D_generators(generators:List[RNG], Nsamples:int):
        """
        Ploteo 3D de los tres generadores para chequear generación
        de Hiperplanos.

        Args:
            generators (List[RNG]): Lista de los generadores
            Nsamples (int): Número de muestras
        """
        fig = plt.figure(figsize=(15, 5))  

        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.view_init(elev=50, azim=-60)
        generators[0].plot_3d_distribution(Nsamples=Nsamples, color="#5B7553", ax=ax1)

        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.view_init(elev=50, azim=-60)
        generators[1].plot_3d_distribution(Nsamples=Nsamples, color="#E76D83", ax=ax2)

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.view_init(elev=50, azim=-60)
        generators[2].plot_3d_distribution(Nsamples=Nsamples, color="#DACC3E", ax=ax3)

        plt.tight_layout()
        plt.show()