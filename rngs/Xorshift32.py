from .RNG import RNG
from time import time

class Xorshift(RNG):
    def __init__(self, seed: int = int(time())):
        super().__init__(seed)

    def seed(self, seed:int):
        self.seed = seed

    def next(self) -> int:
        """
        Metódo que implementa el siguiente número de la secuencia en un
        generador xorshift32

        Returns: 
            int: siguiente número en la secuencia
        """
        x = self._seed
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self._seed = x & 0xFFFFFFFF
        return self._seed
    
    def log(self) -> None:
        """
        Metódo que muestra nombre y las variables que se instancian
        en __init__
        """
        print("NOMBRE: Xorshift32")
        print(f"seed: {self._seed}")

    def rand01(self):
        return self.next() /(2 ** 32)
    
    def name(self) -> str:
        """
        Devuelve el nombre del generador

        Returns:
            str: Nombre del generador LCG
        """
        return "Xorshift 32"