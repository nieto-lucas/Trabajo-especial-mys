from typing import List, Dict, Tuple
import numpy as np
from MonteCarlo import MonteCarlo
from constants import INTEGRAL_VAL_D1
from rngs import RNG
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
    def rng_estimation_gaussian_in_hipercube(Nsamples: int, rng: RNG, d: int = 1) -> float:
        """
        Metódo para calcular la estimación con Monte Carlo de la integral de una
        función gaussiana en un hipercubo de dimensión d, para algun rng
        
        Args:
            Nsamples (int): numero de muestras unfiormes por iteracion
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
            Nsamples (int): numero de muestras unfiormes por iteracion
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
            Nsamples (int): numero de muestras unfiormes por iteracion
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
    def compare_muestral_stats(Nsim: int, Nsamples: int, 
                               seed: int, d: int = 1) -> Dict[str, Tuple[float, float]]:
        """
        Metódo para comparar varianza entre muestras de estimaciones con Monte Carlo
        de la integral de una función gaussiana en un hipercubo de dimensiones d,
        para todos los rngs: LCG, Xorshift, MersenneTwister

        Args:
            Nsim (int): numero de simulaciones de Monte Carlo
            Nsamples (int): numero de muestras unfiormes por iteracion
            d (int): dimension del hipercubo para calcular la integral 
        
        Returns:
            Dict[str,Tuple[float,float]]: entradas con (esperanza muestral, 
            varianza muestral), donde las claves se corresponden a los nombres de las 
            clases de rngs: LCG, Xorshift y MersenneTwister 
        """
        # inicialización de los rngs
        rngs = {
            "LCG": LCG(seed),
            "Xorshift": Xorshift(seed),
            "MersenneTwister": MersenneTwister(seed),
        }
        muestral_stats = {}
        
        try:
            for name, rng in rngs.items():
                muestral_result = Utils.rng_muestral_stat_estimation(Nsim=Nsim,
                                                                    Nsamples=Nsamples,
                                                                    rng=rng, d=d)
                muestral_stats[name] = muestral_result
            return muestral_stats
        
        except Exception as e:
            raise e
    

    @staticmethod
    def compare_cuadratic_error(Nsamples: int, Nsim: int, seed: int, d: int = 1):
        """
        Metódo para comparar varianza entre muestras de estimaciones con Monte Carlo
        de la integral de una función gaussiana en un hipercubo de dimensión d,
        para todos los rngs: LCG, Xorshift, MersenneTwister

        Args:
            Nsim (int): numero de simulaciones de Monte Carlo
            Nsamples (int): numero de muestras unfiormes por iteracion
            d (int): dimension del hipercubo para calcular la integral 
        
        Returns:
            Dict[str,float]: entradas con errores cuadraticos medios, 
            donde las claves se corresponden a los nombres de las clases de 
            rngs: LCG, Xorshift y MersenneTwister 
        """
        # inicialización de los rngs
        rngs = {
            "LCG": LCG(seed),
            "Xorshift": Xorshift(seed),
            "MersenneTwister": MersenneTwister(seed),
        }
        cuadratic_errors = {}

        try:
            for name, rng in rngs.items():
                cuadratic_error = Utils.rng_cuadratic_error_estimation(Nsim=Nsim,
                                                                    Nsamples=Nsamples,
                                                                    rng=rng, d=d) 
                cuadratic_errors[name] = cuadratic_error
            return cuadratic_errors

        except Exception as e:
            raise e