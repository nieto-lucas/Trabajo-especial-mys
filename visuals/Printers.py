from constants import INTEGRAL_VAL_D2, INTEGRAL_VAL_D5, INTEGRAL_VAL_D10
from typing import Tuple, Dict

class Printers:
    """
    Clase con funciones para imprimir resultados de forma bonita
    """

    @staticmethod
    def print_estimations_table(rng_estimations, real_value: float) -> None:
        """
        Imprime en una tabla las estimaciones hechas por los RNGs, 
        incluyendo el error absoluto y el error relativo.

        Args:
            rng_estimations (dict): Estimaciones hechas por los RNGs.
            real_value (float): Valor real de la integral.
        """
        # Determinar etiqueta por dimensión
        if real_value == INTEGRAL_VAL_D2:
            label = "2 DIMENSIONES"
        elif real_value == INTEGRAL_VAL_D5:
            label = "5 DIMENSIONES"
        else:
            label = "10 DIMENSIONES"

        total_width = 96
        padding = (total_width - len(label)) // 2
        print("-" * padding + label + "-" * (total_width - padding - len(label)))

        # Encabezado
        print("| {:^20} | {:^15} | {:^15} | {:^15} | {:^15} |".format(
            "Generador", "Estimación", "Valor Real", "Error Absoluto", "Error Relativo"
        ))
        print("|" + "-" * 22 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 17 + "|")

        # Cuerpo
        rng_names = list(rng_estimations.keys())
        middle_index = len(rng_names) // 2

        for i, (name, estimation) in enumerate(rng_estimations.items()):
            error_abs = abs(estimation - real_value)
            error_rel = error_abs / abs(real_value) if real_value != 0 else 0
            real_str = f"{real_value:.6f}" if i == middle_index else ""
            print("| {:^20} | {:^15.6f} | {:^15} | {:^15.6f} | {:^15.6f} |".format(
                name, estimation, real_str, error_abs, error_rel
            ))

        print("-" * total_width + "\n")

    @staticmethod
    def print_timing_table(dimensional_results: Dict[str, Dict[str, float]], Nsamples: int) -> None:
        """
        Printear tabla de comparativa de tiempos para todas las dimensiones.

        Args:
            dimensional_results: Diccionario de resultados para las distintas dimensiones.
            Nsamples (int): Número de muestras por simulación.
        """
        # Ajustes de formato
        total_width = 60
        dim_width = 15
        gen_width = 20
        time_width = 15

        # Línea superior con info centrada
        print("-" * total_width)
        sim_info = f"{Nsamples} muestras"
        padding = (total_width - len(sim_info)) // 2
        print(" " * padding + sim_info)
        print("-" * total_width)

        # Encabezado de tabla
        print("| {:^{dw}} | {:^{gw}} | {:^{tw}} |".format(
            "Dimensión", "Generador", "Tiempo (s)",
            dw=dim_width, gw=gen_width, tw=time_width
        ))
        print("|" + "-" * (dim_width + 2) + "|" + "-" * (gen_width + 2) + "|" + "-" * (time_width + 2) + "|")

        # Filas de contenido
        for dim, results in dimensional_results.items():
            keys = list(results.keys())
            middle_index = len(keys) // 2
            for i, generator in enumerate(keys):
                time_sec = results[generator]
                dim_str = dim if i == middle_index else ""
                print("| {:^{dw}} | {:^{gw}} | {:^{tw}.6f} |".format(
                    dim_str, generator, time_sec,
                    dw=dim_width, gw=gen_width, tw=time_width
                ))
            print("-" * total_width)
    
    @staticmethod
    def print_stats_table(dimensional_results: Dict[str, Dict[str, Dict[str, float]]], Nsamples: int) -> None:
        """
        Imprime tabla de medias y varianzas muestrales para diferentes dimensiones y tamaños de muestra.

        Args:
            dimensional_results (Dict[str,Dict[str,Tuple[float,float]]]): Diccionario con claves como 
            "2 DIMENSIONES", "5 DIMENSIONES", etc. Cada valor es otro diccionario cuyas claves son 
            tamaños de muestra (int) y cuyos valores son diccionarios con claves "media" y "varianza".
            Nsamples (int): Número de muestras por simulación
        """
        # Ajustes de formato
        total_width = 121
        dim_width = 10
        gen_width = 20
        mean_width = 25
        var_width = 25
        ecm_width = 25

        # Línea superior con info centrada
        print("-" * total_width)
        sim_info = f"{Nsamples} muestras"
        padding = (total_width - len(sim_info)) // 2
        print(" " * padding + sim_info)
        print("-" * total_width)

        # Encabezado
        print("| {:^{dw}} | {:^{sw}} | {:^{mw}} | {:^{vw}} | {:^{ew}} |".format(
            "Dimensión", "Generador", "Media", "Varianza", "ECM", 
            dw=dim_width, sw=gen_width, mw=mean_width, vw=var_width, ew=ecm_width
        ))
        print("|" + "-" * (dim_width + 2) + "|" + "-" * (gen_width + 2) +
            "|" + "-" * (mean_width + 2) + "|" + "-" * (var_width + 2) +
            "|" + "-" * (ecm_width + 2) + "|")

        # Filas
        for dim, sample_dict in dimensional_results.items():
            keys = list(sample_dict.keys())
            middle_index = len(keys) // 2
            for i, sample_size in enumerate(sorted(keys)):
                stats = sample_dict[sample_size]
                media = stats["mean"]
                varianza = stats["variance"]
                ecm = stats["ECM"]
                dim_str = dim if i == middle_index else ""
                print("| {:^{dw}} | {:^{sw}} | {:^{mw}.10f} | {:^{vw}.10f} | {:^{ew}.10f} |".format(
                    dim_str, sample_size, media, varianza, ecm,
                    dw=dim_width, sw=gen_width, mw=mean_width, vw=var_width, ew=ecm_width
                ))
            print("-" * total_width)

    @staticmethod
    def print_testKS_results(rng: str, test_results: Tuple[float, float], alpha: float) -> None:
        """
        Imprime los resultados de realizar el test de Kolmogorov-Smirnov

        Args:
            rng (str): Nombre del RNG
            test_results (Tuple[float, float]): Lista que contiene resultados como:
            - Estadístico D
            - p_valor
            alpha (float): Número de rechazo
        """
        total_length = 70
        title = f"🤝TEST DE KOLMOGOROV-SMIRNOV🔒 - {rng}"
        dash_len = total_length - len(title)
        dashes_left = dash_len // 2
        dashes_right = dash_len - dashes_left
        print("-" * dashes_left + title + "-" * dashes_right)
        print()
        print(f"🧐 D estadístico: {round(test_results[0], 4)}")
        print(f"☝️ p-valor obtenido: {round(test_results[1], 4)}")
        if test_results[1] > alpha:
            print(f"😲☝️ Como {test_results[1]} > {alpha}:")
            print("\t 😒 No hay evidencia suficiente para rechazar Ho")
        else:
            print(f"😲☝️ Como {test_results[1]} <= {alpha}:")
            print(f"\t 🔴 Se rechaza Ho con una confianza del {100 * (1 - alpha)}%")

    def print_testKS_scipy(rng:str, test_results: Tuple[float, float], alpha:float):
        """
        Imprime los resultados de realizar el test de Kolmogorov-Smirnov con scypi

        Args:
            rng (str): Nombre del RNG
            test_results (Tuple[float, float]): Lista que contiene resultados como:
            - Estadístico D
            - p_valor
            alpha (float): Número de rechazo
        """
        print("\n---- (SCIPY🐍) ----\n")
        print(f"🧐 D estadístico: {round(test_results[0], 4)}")
        print(f"☝️ p-valor obtenido: {round(test_results[1], 4)}")
        if test_results[1] > alpha:
            print(f"😲☝️ Como {test_results[1]} > {alpha}:")
            print("\t 😒 No hay evidencia suficiente para rechazar Ho")
        else:
            print(f"😲☝️ Como {test_results[1]} <= {alpha}:")
            print(f"\t 🔴 Se rechaza Ho con una confianza del {100 * (1 - alpha)}%")