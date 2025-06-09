from random import choice
from MonteCarlo import MonteCarlo
import numpy as np
from Utils import Utils
from typing import List, Dict, Tuple
from rngs.RNG import RNG
from rngs.LCG import LCG
from rngs.MersenneTwister import MersenneTwister
from rngs.Xorshift32 import Xorshift
import matplotlib.pyplot as plt
import seaborn as sns
from constants import INTEGRAL_VAL_D1, TWO_DIMENSIONS, TEN_DIMENSIONS, FIVE_DIMENSIONS

class Plotters:

    @staticmethod
    def timing_table(dimensional_results):
        """
        Grafica una comparación de tiempos de ejecución por generador para distintas dimensiones.

        Args:
            dimensional_results (dict): Diccionario con el nombre de la dimensión como clave
                                        y un dict de {generador: tiempo} como valor.
        """
        #Configuración del entorno inicial
        sns.set_theme()
        pallete = sns.color_palette("rocket")

        # Crear figura con un subplot por cada dimensión
        fig, axes = plt.subplots(1, len(dimensional_results), figsize=(5 * len(dimensional_results), 5))

        # Si solo hay un subplot, lo convertimos en lista para iterar consistentemente
        if len(dimensional_results) == 1:
            axes = [axes]

        for ax, (label, result) in zip(axes, dimensional_results.items()):
            generadores = list(result.keys())
            tiempos = list(result.values())

            color = choice(pallete)

            bars = sns.barplot(x=generadores, y=tiempos, ax=ax, color=color)
            ax.set_title(f"Comparación de tiempos\n{label}")
            ax.set_ylabel("Tiempo (s)")
            ax.set_xlabel("Generador")
            ax.set_ylim(0, max(tiempos) * 1.2)

            # Agregar etiquetas arriba de las barras
            for container in bars.containers:
                bars.bar_label(container, fmt='%.6f', label_type='edge', padding=3)

        plt.tight_layout()
        plt.show()
        plt.rcdefaults()        # Restablece parámetros de matplotlib

    @staticmethod    
    def generators_3D(generators: List[RNG], Nsamples: int):
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

    def gaussian_estimation_3D(Nsamples: int, rng: RNG) -> None:
        """
        Ploteo 3D de la estimación con Monte Carlo de la integral
        de una función gaussiana de de dos variables en un 
        cubo [0,1)x[0,1).

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
        ax.set_title(rf"Estimación de Monte Carlo para $\mathcal{{I}}_{{2}}$ ≈ {integral_aprox:.4f}")
        plt.show()

    def gaussian_estimation_Ndim(Nsamples: int, seed: int) -> None:
        """
        Grafica para mostrar como la estimación de Monte Carlo de la
        función gaussiana en un hipercubo de dimensión d estima de mejor
        manera con mayor num de muestras.

        Args:
            Nsamples (int): numero de muestras uniformes.
            seed (int): valor fijo para comparar generadores
        """

        rngs = {
            "LCG": LCG(seed),
            "Xorshift": Xorshift(seed),
            "MersenneTwister": MersenneTwister(seed),
        }
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        for ax, (name, rng) in zip(axes, rngs.items()):
            for d in [TWO_DIMENSIONS, FIVE_DIMENSIONS, TEN_DIMENSIONS]:
                exact_value = INTEGRAL_VAL_D1 ** d
                integral_estims = MonteCarlo.get_integral_per_iteration(
                    Nsamples=Nsamples,
                    g=Utils.gaussian_function,
                    rng=rng
                )
                integral_estims = np.array(integral_estims) ** d
                ax.plot(integral_estims, label=rf"$d={d}$")
                ax.axhline(exact_value, linestyle='--', color='red', linewidth=2)

            ax.set_title(f"RNG: {name}")
            ax.set_xlabel("Número de muestras")
            ax.grid(True)
            ax.legend(title="Dimensión")

        axes[0].set_ylabel("Estimación de la integral")
        fig.suptitle(f"Estimación de Monte Carlo con {Nsamples} muestras", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()