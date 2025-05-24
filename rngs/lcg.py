from .rng import RNG
#from ..constants import LCG_A, LCG_C, LCG_M
from time import time

class LCG(RNG):
    def __init__(self, seed: int = int(time())):
        super().__init__(seed)
        self._a = 16_807
        self._c = 0
        self._m = (2 ** 31) - 1
    
    def next(self) -> int:
        """
        Metódo que implementa el siguiente número de la secuencia en un
        generador congruencial lineal múltiplicativo

        Returns:
            int: siguiente número en la secuencia: (a*s) % m
        """
        self._seed = (self._a* self._seed) % self._m
        return self._seed
    
    def log(self) -> None:
        """
        Metódo que muestra nombre y las variables que se instancian
        en __init__
        """
        print("NOMBRE: Generador congruencial lineal multiplicativo")
        print(f"a: {self._a}")
        print(f"c: {self._c}")
        print(f"m: {self._m}")
        print(f"seed: {self._seed}")
    
    def rand01(self):
        return self.next() /(2 ** 31)