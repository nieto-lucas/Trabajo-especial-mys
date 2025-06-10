from typing import Dict, Tuple, List
from Utils import Utils
from rngs.Xorshift32 import Xorshift
from rngs.MersenneTwister import MersenneTwister
from rngs.LCG import LCG

class Compare:
    """
    Clase para hacer comparaciones entre rngs para los resultados de estimaciones
    de Monte Carlo de una función gaussiana en un hipercubo de dimensión d.
    """

    @staticmethod
    def muestral_stats(Nsim: int, Nsamples: int, 
                        seed: int, d: int = 1) -> Dict[str, Dict[str, float]]:
        """
        Metódo para comparar varianza entre muestras de estimaciones con Monte Carlo
        de la integral de una función gaussiana en un hipercubo de dimensiones d,
        para todos los rngs: LCG, Xorshift, MersenneTwister

        Args:
            Nsim (int): numero de simulaciones de Monte Carlo
            Nsamples (int): numero de muestras uniformes por iteracion
            seed (int): valor fijo para comparar generadores
            d (int): dimension del hipercubo para calcular la integral 
        
        Returns:
            (dict): entradas donde las claves se corresponden a los nombres de 
            las clases de rngs (str): LCG, Xorshift y MersenneTwister y el valor las
            la media y varianza muestral de las estimaciones (Tuple[float,float]) 
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
    def time(Nsamples: int, Nsim: int, 
            seed: int, d: int = 1) -> Dict[str, float]: 
        """
        Metódo para comparar tiempo entre muestras de estimaciones con Monte Carlo
        de la integral de una función gaussiana en un hipercubo de dimensión d,
        para todos los rngs: LCG, Xorshift, MersenneTwister

        Args:
            Nsim (int): numero de simulaciones de Monte Carlo
            Nsamples (int): numero de muestras uniformes por iteracion
            seed (int): valor fijo para comparar generadores
            d (int): dimension del hipercubo para calcular la integral 
        
        Returns:
            (dict): entradas donde las claves se corresponden a los nombres de 
            las clases de rngs (str): LCG, Xorshift y MersenneTwister y el valor los
            tiempos de demora entre rngs (float)
        """
        # inicialización de los rngs
        rngs = {
            "LCG": LCG(seed),
            "Xorshift": Xorshift(seed),
            "MersenneTwister": MersenneTwister(seed),
        }
        times = {}

        try:
            for name, rng in rngs.items():
                time = Utils.rng_time_estimation(Nsamples=Nsamples,
                                                Nsim=Nsim,
                                                rng=rng, d=d)
                times[name] = time
            return times

        except Exception as e:
            raise e
        
    @staticmethod
    def gaussian_estimation_per_iter(Nsamples: int, seed: int, 
                                    d: int = 1) -> Dict[str, List[float]]:
        """
        Metódo para comparar estimaciones con Monte Carlo de la integral de una 
        función gaussiana en un hipercubo de dimensión d, por iteración y para todos 
        los rngs: LCG, Xorshift, MersenneTwister

        Args:
            Nsamples (int): numero de muestras uniformes
            seed (int): valor fijo para comparar generadores
            d (int): dimension del hipercubo para calcular la integral
        
        Returns:
            (dict): entradas donde las claves se corresponden a los nombres de 
            las clases de rngs (str): LCG, Xorshift y MersenneTwister y el valor las
            estimaciones por iteración (List[float]) 
        """
        rngs = {
            "LCG": LCG(seed),
            "Xorshift": Xorshift(seed),
            "MersenneTwister": MersenneTwister(seed),
        }
        estimation_per_iter = {}

        try:
            for name, rng in rngs.items():
                estimations = Utils.rng_gaussian_estimation_per_iter(Nsamples=Nsamples,
                                                                    rng=rng, d=d)
                estimation_per_iter[name] = estimations 
            return estimation_per_iter   
        
        except Exception as e:
            raise e