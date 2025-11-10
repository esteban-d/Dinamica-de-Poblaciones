# Dinámica-de-Poblaciones

Proyecto en **Python** para predecir la evolución temporal de una población mediante
**matrices de Leslie**.

## Módulos

- `analysis.py`: 
Módulo principal del proyecto. Permite obtener las métricas más relevantes de la población, entre las cuales se encuentran:

  - **Tasa de crecimiento** (autovalor dominante)
  - **Sensibilidades** y **elasticidades** 
  - **Autovector derecho** asociado a la tasa de crecimiento (configuración estable)
  - **Autovector izquierdo** asociado a la tasa de crecimiento (valores reproductivos)
  - **Valor reproductivo neto** 
  - **Índice de imprimitividad**
 
La función principal es `gather_information`, que recibe dos vectores:

- `f`: tasas de fecundidad
- `p`: tasas de supervivencia
- `method` (opcional): método numérico

Esta función devuelve un objeto `LeslieInformation`, que encapsula los resultados.

### Ejemplo de uso

```python 
#Tasas de fecundidad
f = np.array([0.0, 0.5, 0.0, 2.25, 0.0, 2.0], dtype=float)
#Tasas de supervivencia
p = np.array([0.3, 0.7, 0.9, 0.9, 0.9], dtype=float)

#Computar información
info = gather_information(f, p)

# Mostrar información principal
print(info)

# Acceder a un atributo específico
print(info.imprimitivity_index)
```

Además, el módulo también incluye una función `display_characteristic_functions`, que grafica las funciones utilizadas para aproximar la tasa de crecimiento y sus cotas, además de aproximaciones con los métodos de Newton y bisección.

- `ejemplos.py`:
Cuenta con tres ejemplos de análisis de poblaciones. En todos los casos se grafican los resultados.

### Visualización de un ejemplo

```python 
# Para visualizar el primer ejemplo: 
ejemplo_1()
```

- `miscellaneous.py`: 
Módulo auxiliar de `ejemplos.py`. Incluye funciones que ayudan a visualizar datos.

- `builders.py`:
Contiene las funciones necesarias para computar la información generada por el módulo `analysis.py`. Está estructurado de manera tal que evita realizar
cómputo redundante al calcular múltiples métricas. También se intentó reducir la complejidad computacional lo más posible, por lo que algunas funciones pueden 
resultar contraintuitivas.

- `data.py`:
Contiene la estructura utilizada para almacenar la información de una matriz, es decir, LeslieInformation.

- `methods.py`:
Contiene los métodos numéricos utilizados en el proyecto. Se encuentra Newton, Bisección y un algoritmo híbrido que aún no ha sido integrado en `analysis.py`.

- `testing.py`: 
Funciones dedicadas al debugging.

