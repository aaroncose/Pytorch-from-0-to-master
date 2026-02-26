# PyTorch 0 to 100: Fundamentos y Práctica

Este repositorio contiene una guía práctica y progresiva para aprender **PyTorch**, desde la manipulación básica de tensores hasta la implementación de modelos de aprendizaje automático desde cero.

## 🚀 Contenido del Curso

El núcleo del aprendizaje se encuentra en el notebook `Pytorch_from_0_to_100_Aarón.ipynb`, que cubre:

### 1. Manipulación de Tensores

- **Creación**: Desde datos existentes, formas deseadas (unos, ceros, aleatorios) y por imitación (`rand_like`).
- **Propiedades**: Inspección de dimensiones (`.shape`), tipos de datos (`.dtype`) y ubicación del dispositivo (`.device`).
- **Indexación y Slicing**: Selección básica, máscaras booleanas y técnicas avanzadas como `torch.gather`.

### 2. Autograd: El Corazón de PyTorch

- Configuración de parámetros entrenables con `requires_grad=True`.
- Grafos de computación y funciones de gradiente (`grad_fn`).
- Cálculo de derivadas mediante el paso atrás (`.backward()`).

### 3. Operaciones Matemáticas

- **Element-wise**: Multiplicación punto a punto (`*`).
- **Producto Matricial**: Uso del operador `@` para álgebra lineal.
- **Reducciones**: Cálculo de medias en diferentes dimensiones (`dim=0`, `dim=1`).

### 4. Modelo desde Cero: Regresión Lineal

Implementación completa del ciclo de entrenamiento siguiendo los **5 pasos fundamentales**:

1. **Creación de Datos**: Generación de variables de entrada y etiquetas con ruido.
2. **Definición del Cerebro**: Inicialización de pesos ($W$) y sesgo ($b$).
3. **Predicción (Forward Pass)**: Implementación de la fórmula $y = XW + b$.
4. **Cuantificación del Error**: Cálculo del Error Cuadrático Medio (MSE).
5. **Optimización**: Ciclo de entrenamiento con actualización manual de parámetros mediante el gradiente descendente:
    $$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L$$

## 🛠️ Requisitos

Para ejecutar el código, necesitarás:

- Python 3.x
- PyTorch
- Jupyter Notebook / Google Colab

## 📖 Cómo usarlo

1. Clona el repositorio.
2. Abre `Pytorch_from_0_to_100_Aarón.ipynb` en tu entorno preferido.
3. Ejecuta las celdas secuencialmente para observar el flujo de aprendizaje.

---
*Notas creadas durante el proceso de aprendizaje de PyTorch.*
