import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Datos de entrenamiento: temperaturas en Celsius y sus equivalentes en Fahrenheit
celsius_values = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_values = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Definición de las capas de la red neuronal
layer_hidden1 = tf.keras.layers.Dense(units=3, input_shape=[1], activation='relu')
layer_hidden2 = tf.keras.layers.Dense(units=3, activation='relu')
layer_output = tf.keras.layers.Dense(units=1)

# Creación del modelo secuencial
model = tf.keras.Sequential([layer_hidden1, layer_hidden2, layer_output])

# Compilación del modelo con el optimizador y la función de pérdida
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
)

# Entrenamiento del modelo
print("Comenzando entrenamiento...")
history = model.fit(celsius_values, fahrenheit_values, epochs=1000, verbose=False)
print("Entrenamiento completado")

# Gráfico de la pérdida a lo largo de las épocas
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.plot(history.history["loss"])
plt.show()

# Realizar una predicción
print("Haciendo una predicción...")
prediction = model.predict(np.array([100.0]))
print(f"El resultado es {prediction[0][0]} Fahrenheit")

# Imprimir los pesos de las capas
print("Pesos de las capas internas del modelo:")
for i, layer in enumerate([layer_hidden1, layer_hidden2, layer_output], start=1):
    print(f"Pesos de la capa {i}: {layer.get_weights()}")
