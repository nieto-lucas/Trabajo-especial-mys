from typing import Callable, Tuple, List
from rngs.RNG import RNG
import numpy as np
from numpy.typing import ArrayLike

class MonteCarlo:
    """
    Implementa el método de MonteCarlo
    """

    @staticmethod
    def method(Nsamples: int,
               g: Callable[[float], float],
               rng: RNG,
               Nvars: int) -> float:
        """
        Método de MonteCarlo multivariable

        Args:
            Nsamples (int): Número de muestras
            g (Callable[[float], float]): Función a aplicar
            rng (RNG): objeto de la clase RNG
            Nvars (int): número de variables

        Returns:
            float: Estimación de la esperanza de g sobre un dominio uniforme.
        """
        integral = 0
        for _ in range(Nsamples):
            uniforms = np.array([rng.rand01() for _ in range(Nvars)])
            integral += g(uniforms)
        return integral/Nsamples
    
    @staticmethod
    def get_parcials_method_Nvars(Nsamples: int,
                                g: Callable[[ArrayLike], float],
                                rng: RNG,
                                Nvars: int) -> List[Tuple[List[float], float]]:
        """
        Método que obtiene del metódo de Monte Carlo para Nvars-variable 
        las uniformes generadas por iteración y el resultado de evaluar 
        g(U_1, ..., U_Nvars). 

        Args:
            Nsamples (int): Número de muestras
            g (Callable[[ArrayLike], float]): Función a aplicar
            rng (RNG): objeto de la clase RNG
            Nvars (int): numero de variables a simular

        Returns:
            List[Tuple[List[float],float]]: Lista con las uniformes generadas
            por iteración de Monte Carlo y el resultado de valuar la función en
            en esas uniformes generadas.
        """
        parcial_estims = []
        integral = 0
        for _ in range(Nsamples):
            uniforms = np.array([rng.rand01() for _ in range(Nvars)])
            parcial_result = g(uniforms)
            parcial_estims.append((uniforms, parcial_result))
            integral += parcial_result
        return parcial_estims


    @staticmethod
    def get_estimation_per_iter(Nsamples: int,
                                g: Callable[[float], float],
                                rng: RNG) -> List[float]:
        """
        Método que obtiene del método de Monte Carlo para una variable
        el resultado de (g(U1)+...+g(Un))/n para cada n-esima iteración 
        entre 1 y Nsamples. 
        
        Args:
            Nsamples (int):
            g (Callable[[float],float]): Función a aplicar
            rng (RNG): objeto de la clase RNG.
        
        Returns:
            List[float]: Lista con los resultados de los resultados de
            (g(U1)+...+g(Un))/n para cada n-esima iteración.
        """
        integral_iter = []        
        integral = 0
        for n in range(Nsamples):
            U = rng.rand01()
            integral += g(U)
            integral_iter.append(integral/(n+1))
        return integral_iter

    def get_muestral_stats(Nsamples:int,  Nvars:int,
                rng:RNG, g:Callable[[float], float]) -> Tuple[float, float]:

        U_1 = np.array([rng.rand01() for _ in range(Nvars)])
        media = g(U_1)
        scuad, n = 0, 1
        while n < Nsamples: 
            n += 1
            samples = np.array([rng.rand01() for _ in range(Nvars)])
            g_samples = g(samples)
            media_ant = media
            media = media_ant + (g_samples - media_ant) / n
            scuad = scuad * (1 - 1 /(n-1)) + n*(media - media_ant)**2
        return scuad, media