# Generadores de Números Pseudoaleatorios y Simulación

Este proyecto corresponde al Trabajo Práctico Especial de la materia **Modelos y Simulación 2025**, realizado por *Alvaro Medina* y *Lucas Nieto*. Tiene como objetivo analizar el rendimiento de distintos generadores de números pseudoaleatorios (PRNGs) aplicados a la estimación de una integral definida sobre el hipercubo $[0, 1]^d$ utilizando el método de Monte Carlo.

El análisis completo, incluyendo fundamentos teóricos, resultados, comparaciones y conclusiones, se encuentra disponible en el informe oficial:  
📄 **[Informe completo del proyecto](https://drive.google.com/file/d/1TsGy2va6Ie3k5tnftjMj4L06xeyt7zXa/view?usp=sharing)**

## Descripción del proyecto

Se evaluaron tres generadores representativos:

- **Congruencial Lineal (LCG)**
- **Xorshift32**
- **Mersenne Twister**

Comparando sus resultados en términos de:
- Precisión (Error Cuadrático Medio)
- Varianza de las estimaciones
- Tiempos de ejecución
- Aleatoriedad a través del test de Kolmogorov-Smirnov

## Estructura del repositorio

El análisis se organizó y ejecutó en el archivo `trabajo-especial.ipynb`, que combina resultados teóricos, simulaciones y visualizaciones. Este notebook hace uso de los siguientes módulos Python:

### 📁 `analysis/`
Contiene cálculos y comparaciones entre los generadores.

- `Compare.py`: Funciones para estimar ECM, esperanza, varianza y tiempos de ejecución. Utiliza objetos de la clase `RNG` con la misma semilla para realizar comparaciones justas.

### 📄 `constants.py`
Constantes del experimento, como el valor exacto de la integral, número de muestras, dimensiones, etc.

### 📄 `MonteCarlo.py`
Implementación del método de Monte Carlo para estimar integrales en múltiples dimensiones.

### 📁 `rng/`
Implementaciones de los generadores de números aleatorios estudiados.

- `LCG.py`: Generador Congruencial Lineal.
- `Mersenne Twister.py`: Adaptación de la versión original en C del Mersenne Twister a Python.
- `RNG.py`: Clase base para generadores aleatorios.
- `Xorshift32.py`: Adaptación del generador Xorshift32 desde C, con operaciones en F₂³² aseguradas mediante máscaras.

### 📁 `test/`
Contiene código relacionado al test estadístico de Kolmogorov-Smirnov.

- `Test.py`: Implementación del test.
- `TestHelpers.py`: Funciones auxiliares, como el cálculo del estadístico KS.

### 📄 `Utils.py`
Funciones utilitarias para apoyar las comparaciones entre generadores y los tests.

### 📁 `visuals/`
Herramientas para la visualización de resultados.

- `Plotters.py`: Gráficos comparativos de resultados entre generadores.
- `Printers.py`: Imprime por consola los resultados obtenidos y conclusiones del test de hipótesis.

---

Este trabajo explora cómo distintos RNGs afectan la precisión y confiabilidad de las simulaciones estadísticas, y sirve como base para futuras mejoras en generación de muestras aleatorias en contextos científicos y de ingeniería.