from .RNG import RNG

class MersenneTwister(RNG):
    
    '''
        Constantes propias del método
    '''
    BIT_WIDTH = 32                      # Longitud de palabra en bits (w)
    MT_STATE_SIZE = 624                 # Tamaño del vector de estados (n)
    RECURRENCE_OFFSET = 397             # Offset para la recurrencia (m)
    SEPARATION_INDEX = 31               # Índice de separación (r)

    MATRIX_A = 0x9908B0DF               # Constante A utilizada en el twist
    UPPER_MASK = 0x80000000             # Máscara de bits superiores (primer bit en 1, resto en 0)
    LOWER_MASK = 0x7FFFFFFF             # Máscara de bits inferiores (primer bit en 0, resto en 1)

    # Parámetros de temperado (tempering)
    TEMPERING_SHIFT_U = 11              # Desplazamiento a la derecha para temperado (u)
    TEMPERING_MASK_D = 0xFFFFFFFF       # Máscara D (comúnmente todos los bits en 1)

    TEMPERING_SHIFT_S = 7               # Desplazamiento a la izquierda (s)
    TEMPERING_MASK_B = 0x9D2C5680       # Máscara B

    TEMPERING_SHIFT_T = 15              # Desplazamiento a la izquierda (t)
    TEMPERING_MASK_C = 0xEFC60000       # Máscara C

    TEMPERING_SHIFT_L = 18              # Desplazamiento final a la derecha (l)

    KNUTH_MULTIPLIER = 1812433253       # Multiplicador de Knuth para la inicialización (f)

    def __init__(self, seed_value):
        super().__init__(seed_value)  # Llama al constructor de RNG
        self._mt = [0] * self.MT_STATE_SIZE
        self.index = self.MT_STATE_SIZE
        self.seed(seed_value)

    def seed(self, num):
        """Inicializa el generador con una semilla"""
        num &= 0xFFFFFFFF # Asegurar 32 bits
        self._mt[0] = num
        self.index = self.MT_STATE_SIZE
        for i in range(1, self.MT_STATE_SIZE):
            temp = self.KNUTH_MULTIPLIER * (self._mt[i - 1] ^ (self._mt[i - 1] >> (self.BIT_WIDTH - 2))) + i
            self._mt[i] = temp & 0xFFFFFFFF  # Asegura 32 bits

    def twist(self):
        """Genera los próximos MT_STATE_SIZE valores"""
        for i in range(self.MT_STATE_SIZE):
            x = (self._mt[i] & self.UPPER_MASK) + (self._mt[(i + 1) % self.MT_STATE_SIZE] & self.LOWER_MASK)
            xA = x >> 1
            if x % 2 != 0:
                xA ^= self.MATRIX_A
            self._mt[i] = self._mt[(i + self.RECURRENCE_OFFSET) % self.MT_STATE_SIZE] ^ xA
        self.index = 0

    def extract_number(self):
        """Extrae un número temperado de la secuencia"""
        if self.index >= self.MT_STATE_SIZE:
            self.twist()

        y = self._mt[self.index]
        y ^= (y >> self.TEMPERING_SHIFT_U) & self.TEMPERING_MASK_D
        y ^= (y << self.TEMPERING_SHIFT_S) & self.TEMPERING_MASK_B
        y ^= (y << self.TEMPERING_SHIFT_T) & self.TEMPERING_MASK_C
        y ^= (y >> self.TEMPERING_SHIFT_L)

        self.index += 1
        return y & 0xFFFFFFFF

    def random(self):
        """Devuelve un número de punto flotante en el rango [0, 1)"""
        return self.extract_number() / 2**self.BIT_WIDTH
    
    def next(self) -> int:
        """
        Devuelve el siguiente número entero de 32 bits de la secuencia
        """
        return self.extract_number()
    
    def rand01(self) -> float:
        """
        Devuelve un número flotante en el rango [0, 1)
        """
        return self.extract_number() / 2**self.BIT_WIDTH

    def log(self) -> None:
        """
        Muestra el estado actual del generador (parcial, para depuración)
        """
        print(f"[MersenneTwister] Estado actual:")
        print(f"Index: {self.index}")
        print(f"Seed: {self.get_seed()}")
        print(f"Primeros 5 valores del estado: {self._mt[:5]}")

    def name(self) -> str:
        """
        Devuelve el nombre del generador

        Returns:
            str: Nombre del generador LCG
        """
        return "Mersenne Twister"
