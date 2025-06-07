from random import choice 
import matplotlib.pyplot as plt
import seaborn as sns



class Plotters:

    @staticmethod
    def plot_timing_table(dimensional_results):
        """
        Grafica una comparación de tiempos de ejecución por generador para distintas dimensiones.

        Args:
            dimensional_results (dict): Diccionario con el nombre de la dimensión como clave
                                        y un dict de {generador: tiempo} como valor.
        """
        #Configuración del entorno inicial
        sns.set_theme()
        pallete = sns.color_palette("hls", 8)

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
