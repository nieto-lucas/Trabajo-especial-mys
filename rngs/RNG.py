from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class RNG(ABC):
    """
    Implementa una clase abstracta para todos los RNG's que probaremos
    """
    def __init__(self, seed: int):
        if seed == 0:
            raise Exception("ERROR: el estado de seed no puede ser 0")
        
        self._seed = seed
    
    @abstractmethod
    def set_seed(self, seed:int):
        """
        Inicializa el valor de la seed para un generador

        Args:
            seed (int): seed a cambiar
        """
        pass


    @abstractmethod
    def next(self) -> int:
        pass

    @abstractmethod
    def log(self) -> None:
        pass

    def get_seed(self) -> int:
        """
        Obtiene la seed de RNG útilizado

        Returns:
            int: seed del RNG
        """
        return self._seed
    
    @abstractmethod
    def rand01(self) -> float:
        """
        Metódo next normalizado a valores uniformes en [0, 1)
        
        Returns:
            float: siguiente número de la secuencia normalizado en [0, 1) 
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Devuelve el nombre del generador
        """
        pass

    def plot_3d_distribution(self, Nsamples: int, color: str, ax=None):
        """
        Plotea la distribución 3D ordenada de los números generados por el RNG.

        Args:
            Nsamples (int): Número de muestras a generar para la distribución. 
                            Debe ser al menos 3 para poder formar los puntos 3D.
            color (str): Color utilizado para los puntos en el gráfico (ejemplo: '#FF0000').
            ax (matplotlib.axes._subplots.Axes3DSubplot, optional): 
                Objeto Axes3D donde se dibujará el gráfico. 
                Si no se proporciona, la función creará una nueva figura y eje.
                Defaults to None.

        Raises:
            ValueError: Si `Nsamples` es menor que 3, ya que no se pueden formar puntos 3D.
        """
        if Nsamples < 3:
            raise ValueError("Se necesitan al menos 3 muestras.")

        values = [self.rand01() for _ in range(Nsamples)]
        x_values = values[:-2]
        y_values = values[1:-1]
        z_values = values[2:]

        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x_values, y_values, z_values, c=color, marker='o', s=10, alpha=0.6)
        ax.set_title(f"Distribución 3D de puntos de {self.name()}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if ax is None:
            plt.show()