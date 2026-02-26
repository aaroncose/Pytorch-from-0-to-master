# PyTorch 0 to 100: Fundamentos y Práctica

Guía práctica y progresiva para aprender **PyTorch**, desde la manipulación básica de tensores hasta la implementación de modelos de aprendizaje automático desde cero.

## 🛠️ Requisitos

- Python 3.x
- PyTorch (`pip install torch`)
- Jupyter Notebook o Google Colab

---

## 📖 Manual de Uso

El notebook está diseñado para ejecutarse **celda a celda de arriba a abajo**. Cada sección construye sobre la anterior.

### ① Crear un Tensor desde datos

```python
import torch

data = [[3, 2, 1], [4, 5, 2]]
my_tensor = torch.tensor(data)
print(my_tensor)
```

> **Cuándo usarlo**: Tienes tus propios datos (una lista, un array) y los quieres convertir al formato de PyTorch.

---

### ② Crear Tensores con formas predefinidas

```python
shape = (2, 3)
ones   = torch.ones(shape)    # Tensor de unos
zeros  = torch.zeros(shape)   # Tensor de ceros
random = torch.rand(shape)    # Tensor aleatorio entre 0 y 1
```

> **Cuándo usarlo**: Necesitas inicializar parámetros o crear datos de prueba rápidamente sin datos reales.

---

### ③ Inspeccionar un Tensor

```python
tensor = torch.randn(2, 9)
print(tensor.shape)   # → torch.Size([2, 9])
print(tensor.dtype)   # → torch.float32
print(tensor.device)  # → cpu
```

> **Cuándo usarlo**: Siempre que tengas un error de dimensiones. `.shape` es tu herramienta de depuración nº1.

---

### ④ Marcar un parámetro como entrenable (Autograd)

```python
W = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
```

> **Cuándo usarlo**: Cuando esa variable es un peso o sesgo que el modelo necesita aprender. Sin `requires_grad=True`, PyTorch no calculará su gradiente.

---

### ⑤ Calcular gradientes

```python
# 1. Define la operación
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
z = a * b  # z = 6.0

# 2. Dispara el cálculo de gradientes
z.backward()

# 3. Lee el gradiente de cada variable
print(a.grad)  # dz/da = b = 3.0
print(b.grad)  # dz/db = a = 2.0
```

> **Cuándo usarlo**: Para entender cuánto contribuye cada parámetro al error. Es la base del aprendizaje automático.

---

### ⑥ Operaciones matriciales

```python
# Multiplicación elemento a elemento
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[10, 20], [30, 40]])
print(a * b)  # [[10, 40], [90, 160]]

# Producto matricial (la más común en deep learning)
m1 = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
m2 = torch.tensor([[7, 8], [9, 10], [11, 12]])  # Shape: (3, 2)
print(m1 @ m2)  # Shape resultante: (2, 2)
```

> **Regla clave**: Para `@`, las columnas de la primera matriz deben ser iguales a las filas de la segunda.

---

### ⑦ Reducciones por dimensión

```python
# Imagina: 2 estudiantes, 3 asignaciones → shape (2, 3)
scores = torch.tensor([[10., 20., 30.], [5., 10., 15.]])

# Media por asignación (colapsa filas → dim=0)
print(scores.mean(dim=0))  # → [7.5, 15.0, 22.5]

# Media por estudiante (colapsa columnas → dim=1)
print(scores.mean(dim=1))  # → [20.0, 10.0]
```

> **Regla mnemotécnica**: `dim=0` → colapsa las **filas** (baja verticalmente). `dim=1` → colapsa las **columnas** (barre horizontalmente).

---

### ⑧ Indexación y selección

```python
x = torch.arange(12).reshape(3, 4)

x[:, 2]      # Toda la columna índice 2
x[:, -1]     # Última columna
x[0:2, 1:3]  # Sub-tensor: filas 0-1, columnas 1-2
x[x > 7]     # Máscara booleana: solo valores > 7
```

---

### ⑨ El bucle de entrenamiento completo (5 pasos)

Este es el patrón que se repite en todo proyecto de PyTorch:

```python
learning_rate, epochs = 0.01, 100
W = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

for epoch in range(epochs):
    # 1. FORWARD PASS → predecir
    y_hat = X @ W + b

    # 2. CALCULAR PÉRDIDA (MSE)
    loss = torch.mean((y_hat - y_true) ** 2)

    # 3. BACKWARD → calcular gradientes
    loss.backward()

    # 4. ACTUALIZAR PARÁMETROS
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    # 5. LIMPIAR GRADIENTES (¡obligatorio antes del siguiente epoch!)
    W.grad.zero_()
    b.grad.zero_()
```

> ⚠️ **Error más común**: Olvidarse de `W.grad.zero_()`. Si no limpias los gradientes, se **acumulan** de un epoch al siguiente y el modelo no aprende correctamente.

---

## 🗂️ Contenido del Repositorio

| Archivo | Descripción |
|---|---|
| `Pytorch_from_0_to_100_Aarón.ipynb` | Notebook principal con todo el código y explicaciones |
| `README.md` | Este manual |

---
*Notas creadas durante el proceso de aprendizaje de PyTorch.*
