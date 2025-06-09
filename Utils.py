from typing import Tuple, List
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
        
        estimation = MonteCarlo.method(Nsamples=Nsamples, 
                                      g=Utils.gaussian_function,
                                      rng=rng
                                      ) ** d
        return estimation

    @staticmethod
    def rng_muestral_stat_estimation(Nsim: int, Nsamples: int, 
                                     rng: RNG, d: int = 1) -> Tuple[float, float]:
        """
        Metódo para calcular la esperanza muestral y la varianza muestral 
        correjida entre muestras de estimaciones con Monte Carlo de la integral de 
        una función gaussiana en un hipercubo de dimensiones d, para algun rng

        Args:
            Nsim (int): numero de simulaciones de Monte Carlo
            Nsamples (int): numero de muestras uniformes por iteracion
            rng (RNG): objeto de la clase RNG para obtener uniformes
            d (int): dimension del hipercubo para calcular la integral

        Returns:
            Tuple[float,float]: (esperanza muestral, varianza muestral) de la 
            estimación respecto al resultado exacto de la integral de la función 
            gausiana en un hipercubo de dim d
        """
        if Nsim <= 0:
            raise Exception("Error: la cantidad de simulaciones debe ser mayor a 0")

        try:
            media = Utils.rng_estimation_gaussian_in_hipercube(Nsamples=Nsamples,
                                                                rng=rng,
                                                                d=d)
            scuad, n = 0, 1
            # cada rng recolecta muestras
            while n < Nsim: 
                n += 1
                estim = Utils.rng_estimation_gaussian_in_hipercube(Nsamples=Nsamples, 
                                                                    rng=rng, 
                                                                    d=d)
                media_ant = media
                media = media_ant + (estim - media_ant) / n
                scuad = scuad * (1 - 1 /(n-1)) + n*(media - media_ant)**2
            return (media, scuad)
        
        except Exception as e:
            raise e
    
    @staticmethod
    def rng_cuadratic_error_estimation(Nsim: int, Nsamples: int, 
                                        rng: RNG, d: int = 1) -> float:
        """
        Metódo para calcular el error cuadratico medio entre muestras de 
        estimaciones con Monte Carlo de la integral de una función gaussiana 
        en un hipercubo de dimensiones d, para algun rng

        Args:
            Nsim (int): numero de simulaciones de Monte Carlo
            Nsamples (int): numero de muestras uniformes por iteracion
            rng (RNG): objeto de la clase RNG para obtener uniformes
            d (int): dimension del hipercubo para calcular la integral 

        Returns:
            float: Error cuadratico medio de la estimación respecto al resultado
            exacto de la integral de la función gausiana en un hipercubo de dim d
        """
        if Nsim <= 0:
            raise Exception("Error: la cantidad de simulaciones debe ser mayor a 0")

        try:
            exact_integral_val = INTEGRAL_VAL_D1**d 
            cuadratic_error = 0
            for _ in range(Nsim):
                estim = Utils.rng_estimation_gaussian_in_hipercube(Nsamples=Nsamples, 
                                                                    rng=rng, 
                                                                    d=d)
                cuadratic_error += (estim - exact_integral_val)**2
            return cuadratic_error/ Nsim
        
        except Exception as e:
            raise e 
        
    @staticmethod
    def rng_time_estimation(Nsim: int, Nsamples: int, 
                            rng: RNG, d: int = 1) -> float:
        """
        Metódo para obtener el tiempo entre muestras de estimaciones con Monte Carlo 
        de la integral de una función gaussiana en un hipercubo de dimensiones d, 
        para algun rng

        Args:
            Nsim (int): numero de simulaciones de Monte Carlo
            Nsamples (int): numero de muestras uniformes por iteracion
            rng (RNG): objeto de la clase RNG para obtener uniformes
            d (int): dimension del hipercubo para calcular la integral 

        Returns:
            float: tiempo de demora de Nsim-estimaciones de la integral
        """
        try:
            start = time()
            for _ in range(Nsim):
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