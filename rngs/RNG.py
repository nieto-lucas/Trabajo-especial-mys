from abc import ABC, abstractmethod

class RNG(ABC):
    """
    Implementa una clase abstracta para todos los RNG's que probaremos
    """
    def __init__(self, seed: int):
        if seed == 0:
            raise Exception("ERROR: el estado de seed no puede ser 0")
        
        self._seed = seed
    
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