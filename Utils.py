from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from utils.MonteCarlo import MonteCarlo
from constants import INTEGRAL_VAL_D1
from rngs.RNG import RNG
from rngs.Xorshift32 import Xorshift
from rngs.MersenneTwister import MersenneTwister
from rngs.LCG import LCG
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
    def rng_compare_time(Nsim: int, Nsamples: int, 
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
    def compare_muestral_stats(Nsim: int, Nsamples: int, 
                               seed: int, d: int = 1) -> Dict[str, Tuple[float, float]]:
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
    def compare_cuadratic_error(Nsamples: int, Nsim: int, 
                                seed: int, d: int = 1) -> Dict[str, float]:
        """
        Metódo para comparar varianza entre muestras de estimaciones con Monte Carlo
        de la integral de una función gaussiana en un hipercubo de dimensión d,
        para todos los rngs: LCG, Xorshift, MersenneTwister

        Args:
            Nsim (int): numero de simulaciones de Monte Carlo
            Nsamples (int): numero de muestras uniformes por iteracion
            seed (int): valor fijo para comparar generadores
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
        
    @staticmethod
    def compare_time(Nsamples: int, Nsim: int, 
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
        times = {}

        try:
            for name, rng in rngs.items():
                time = Utils.rng_compare_time(Nsamples=Nsamples,
                                            Nsim=Nsim,
                                            rng=rng, d=d)
                times[name] = time
            return times

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

    def plot_3D_gaussian_estimation(Nsamples: int, rng: RNG) -> None:
        """
        Ploteo 3D de la estimación con Monte Carlo de la integral
        de una función gaussiana de de dos variables en un 
        cubo [0,1)x[0,1)x[0,1).

        Args:
            Nsamples (int): numero de muestras uniformes por iteracion
            rng (RNG): objeto de la clase RNG para obtener uniformes
        """
        samples = MonteCarlo.get_parcials_method_Nvars(Nsamples=Nsamples, 
                                                  g=Utils.gaussian_func_multivar, 
                                                  rng=rng, 
                                                  Nvars=2)
        samples_array = np.array(samples, dtype=object)
        coords = np.stack(samples_array[:, 0])
        z_samples = np.array(samples_array[:, 1], dtype=float)
        x_samples = coords[:, 0]
        y_samples = coords[:, 1]

        integral_aprox = np.mean(z_samples)

        X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        Z = np.exp(-X**2-Y**2) 

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.4)

        dx = dy = 1 / len(samples) ** 0.7
        for x, y, z in zip(x_samples, y_samples, z_samples):
            ax.bar3d(x, y, 0, dx, dy, z, color='red', alpha=0.3)

        ax.view_init(elev=30, azim=60)
        ax.set_title(f"Estimación de integral ≈ {integral_aprox:.4f}")
        plt.show()
