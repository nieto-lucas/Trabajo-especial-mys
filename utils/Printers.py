from constants import INTEGRAL_VAL_D2, INTEGRAL_VAL_D5, INTEGRAL_VAL_D10


class Printers:
    """
    Clase con funciones para imprimir resultados de forma bonita
    """

    @staticmethod
    def print_estimations_table(rng_estimations: dict, real_value: float):
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

        total_width = 97
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
