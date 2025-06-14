from typing import Dict, List
import numpy as np
from MonteCarlo import MonteCarlo
from constants import INTEGRAL_VAL_D1
from rngs.RNG import RNG
from time import time

class Utils:
    """
    Clase con funciones utiles
    """
    
    @staticmethod
    def gaussian_func_multivar(Xs: np.ndarray) -> float:
        """
        Función multivariable que se usa para estimar el valor de la integral 
        con metódo de Monte Carlo.

        Args:
            Xs(np.adarray): valor con el que se inicializa la función gaussiana 
        
        Returns: 
            float: retorna el valor de la función gaussiana valuada en las variables
        """
        return np.exp(-np.sum(Xs**2))
    
    @staticmethod
    def rng_estimation_gaussian_in_hipercube(Nsamples: int, rng: RNG, d: int = 1) -> float:
        """
        Metódo para calcular la estimación con Monte Carlo de la integral de una
        función gaussiana en un hipercubo de dimensión d, para algun rng
        
        Args:
            Nsamples (int): numero de muestras uniformes por iteracion
            rng (RNG): objeto de la clase RNG para obtener uniformes
            d (int): dimension del hipercubo para calcular la integral

        Returns:
            float: estimación con metódo de Monte Carlo de la integral de
            una función gaussiana en un hipercubo de dimensión d.
        """
        if d < 1:
            raise Exception("Error: la dimensión debe ser mayor a 1")
        
        estimation = MonteCarlo.method(
                            Nsamples=Nsamples, 
                            g=Utils.gaussian_func_multivar,
                            rng=rng,
                            Nvars=d)
        return estimation

    @staticmethod
    def rng_muestral_stats_estimation_hipercube(Nsamples: int, rng: RNG, d: int = 1) -> float:
        if d < 1:
            raise Exception("Error: la dimensión debe ser mayor a 1")
        
        var, mean = MonteCarlo.get_muestral_stats(
                                Nsamples=Nsamples,
                                Nvars=d,
                                rng=rng,
                                g=Utils.gaussian_func_multivar)
        var /= Nsamples
        ecm = var + (mean - INTEGRAL_VAL_D1 ** d) ** 2
        results = {
            "variance": var,
            "mean": mean,
            "ECM": ecm
        }
        return results

    @staticmethod
    def rng_time_estimation(Nsamples: int, rng: RNG, d: int = 1) -> float:
        """
        Metódo para obtener el tiempo entre muestras de estimaciones con Monte Carlo 
        de la integral de una función gaussiana en un hipercubo de dimensiones d, 
        para algun rng

        Args:
            Nsamples (int): numero de muestras uniformes por iteracion
            rng (RNG): objeto de la clase RNG para obtener uniformes
            d (int): dimension del hipercubo para calcular la integral 

        Returns:
            float: tiempo de demora de estimar la integral
        """
        try:
            start = time()
            Utils.rng_estimation_gaussian_in_hipercube(Nsamples=Nsamples,
                                                rng=rng, 
                                                d=d)
            end = time()
            return end - start

        except Exception as e:
            raise e
        
    @staticmethod
    def rng_gaussian_estimation_per_iter(Nsamples: int, rng: RNG,
                                        d: int = 1) -> List[float]:
        """ 
        Metódo para obtener el tiempo entre muestras de estimaciones con Monte Carlo 
        de la integral de una función gaussiana en un hipercubo de dimensiones d, 
        para algun rng

        Args:
            Nsamples (int): numero de muestras uniformes
            rng (RNG): objeto de la clase RNG para obtener uniformes
            d (int): dimension del hipercubo para calcular la integral 

        Returns:
            List[float]: estimaciones con Monte Carlo de la integral de una
            función gaussiana por iteración. 
        """
        try:
            estimation_per_iter = MonteCarlo.get_estimation_per_iter(
                Nsamples=Nsamples,
                g=Utils.gaussian_function,
                rng=rng
            )
            estimation_per_iter = np.array(estimation_per_iter) ** d
            return estimation_per_iter.tolist()
        
        except Exception as e:
            raise e