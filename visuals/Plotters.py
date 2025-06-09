from random import choice
from MonteCarlo import MonteCarlo
import numpy as np
from Utils import Utils
from typing import List, Dict, Optional, Callable
from rngs.RNG import RNG
import matplotlib.pyplot as plt
import seaborn as sns
from constants import INTEGRAL_VAL_D1
from matplotlib.ticker import FuncFormatter

class Plotters:
    """ 
    """

    @staticmethod
    def _barplot_common(dim_res: Dict[int, Dict[str, float]],
                        ylabel: str,
                        title_prefix: str,
                        bar_label_formatter: Optional[Callable[[float], str]] = None,
                        yaxis_formatter: Optional[Callable[[float, int], str]] = None) -> None:
        """
        Función general para hacer graficas de barras de resultados asociados a RNGs
        agrupados por dimensión.

        Args:
            dim_res (dict): Diccionario con clave dimensión (int), 
            de valor un dict con clave generador (str) y con valor (float) asociado
            al RNG para esa dimensión. 
            ylabel (str): comentario del eje 'y'
            title_prefix (str): nombre global a los graficos de barras.
            bar_label_formatter (Callable[[float], str]): agrega formato los datos de
            cada barra.
            yaxis_formatter (Callable[[float, int], str]): agrega formato al eje 'y'
        """
        sns.set_theme()
        palette = sns.color_palette("rocket")

        fig, axes = plt.subplots(1, len(dim_res), figsize=(5 * len(dim_res), 5))
        if len(dim_res) == 1:
            axes = [axes]

        for ax, (label, result) in zip(axes, dim_res.items()):
            labels = list(result.keys())
            values = list(result.values())
            color = choice(palette)

            bars = sns.barplot(x=labels, y=values, ax=ax, color=color)
            ax.set_title(f"DIMENSIONES {label}")
            ax.set_xlabel("Generador")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(values) * 1.2)

            # Formateo del eje Y
            if yaxis_formatter:
                ax.yaxis.set_major_formatter(FuncFormatter(yaxis_formatter))

            # Formateo de etiquetas arriba de las barras
            if bar_label_formatter:
                for container in bars.containers:
                    labels = [bar_label_formatter(val) for val in container.datavalues]
                    bars.bar_label(container, labels=labels, label_type='edge', padding=3)
            else:
                for container in bars.containers:
                    bars.bar_label(container, fmt='%.6f', label_type='edge', padding=3)

        fig.suptitle(f"{title_prefix}", fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.rcdefaults()

    @staticmethod
    def time_bars(dim_res: Dict[int, Dict[str, float]]) -> None:
        """ 
        Grafica barras del tiempo de estimaciones de integral con Monte Carlo con 
        varios RNGs agrupados por dimensión.

        Args:
            dim_res (dict): Diccionario con clave dimensión (int), 
            de valor un dict con clave generador (str) y con valor tiempo de
            demora de las esimaciones de Monte Carlo (float). 
        """
        Plotters._barplot_common(dim_res, 
                                 ylabel="Tiempo (s)", 
                                 title_prefix="Comparación de tiempos")

    @staticmethod
    def variance_bars(dim_res: Dict[int, Dict[str, List[float]]]) -> None:
        """ 
        Grafica barras de la varianza de estimaciones de integral con Monte Carlo con 
        varios RNGs agrupados por dimensión.
        
        Args:
            dim_res (dict): Diccionario con clave dimensión (int), 
            de valor un dict con clave generador (str) y con valor varinza entre
            estimaciones de Monte Carlo (float). 
        """
        Plotters._barplot_common(dim_res,
                                ylabel="Varianza",
                                title_prefix="Comparación de varianzas",
                                yaxis_formatter=lambda x, _: f'{x:.1e}',
                                bar_label_formatter=lambda x: f'{x:.1e}')

    def cuadratic_error_bars(dim_res: Dict[int, Dict[str, List[float]]]) -> None:
        """ 
        Grafica barras del ECM de estimaciones de integral con Monte Carlo con 
        varios RNGs agrupados por dimensión.
        
        Args:
            dim_res (dict): Diccionario con clave dimensión (int), 
            de valor un dict con clave generador (str) y con valor ECM entre
            estimaciones de Monte Carlo (float).
        """
        Plotters._barplot_common(dim_res, 
                                 ylabel="ECM", 
                                 title_prefix="Comparación de error cuadratico medio (ECM)")

    @staticmethod    
    def generators_3D(generators: List[RNG], Nsamples: int) -> None:
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

    def gaussian_estimations_Ndim(dim_res: Dict[int, Dict[str, List[float]]]) -> None:
        """
        Grafica para mostrar como la estimación de Monte Carlo de la
        función gaussiana en un hipercubo de dimensión d estima de mejor
        manera con mayor num de muestras.

        Args:
            dim_res (dict): Diccionario con clave dimensión (int), 
            de valor un dict con clave generador (str) y con valor la lista 
            de integrales por iteración del método de Monte Carlo para la función
            gaussiana en un hipercubo de dim d (list[float]).
        """

        rng_names = list(next(iter(dim_res.values())).keys())
        dims = list(dim_res.keys())
        color_map = {d: c for d, c in zip(dims, sns.color_palette("tab10", n_colors=len(dims)))}

        fig, axes = plt.subplots(1, len(rng_names), figsize=(6 * len(rng_names), 5), sharey=True)

        if len(rng_names) == 1:
            axes = [axes]

        for ax, rng_name in zip(axes, rng_names):
            for d in dims:
                estimations = dim_res[d][rng_name]
                exact_value = INTEGRAL_VAL_D1 ** d
                color = color_map[d]

                ax.plot(estimations, label=rf"$d={d}$", color=color)
                ax.axhline(exact_value, linestyle="--", color=color, alpha=0.8, linewidth=1.2)

            ax.set_title(f"RNG: {rng_name}")
            ax.set_xlabel("Número de muestras")
            ax.grid(True)
            ax.legend(title="Dimensión")

        axes[0].set_ylabel("Estimación de la integral")
        fig.suptitle("Estimación de Monte Carlo por generador y dimensión", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()