from rngs.RNG import RNG
from rngs import *
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

class Test:
    @staticmethod
    def test_Kolmogorov_Smirnov(rng:RNG, Nsamples:int):
        #Set para no generar muestras repetidas
        samples = set(rng.rand01() for _ in range(Nsamples))

        #Ordeno las muestras
        x_samples = np.concatenate(([0], np.sort(list(samples)), [1]))

        n = len(samples)

        #Calculo la empírica
        Fe_samples = np.concatenate(([0], np.arange(1, n+1, 1) / n,[1]))

        # Distribución Uniforme Real
        x_values = np.linspace(0,1, Nsamples)
        y_values = stats.uniform.cdf(x_values)
        
        sns.lineplot(x=x_values, y=y_values, color='#5B7553', label="F(x)")
        plt.step(x=x_samples, y=Fe_samples, color='#E76D83', label="Fe(X)")
        plt.title(f"Fe(x) vs F(x) utilizando {rng.name()}")
        plt.legend()
        plt.show()

