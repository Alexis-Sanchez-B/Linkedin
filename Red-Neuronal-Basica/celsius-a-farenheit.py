import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;


celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float);
farenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float);

capa = tf.keras.layers.Dense(units=1, input_shape=[1]); #capa de entrada con una neurona
modelo = tf.keras.Sequential([capa]); #modelo secuencial con una capa
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
    """optimizador y perdida, permite saber a la red ajustar los pesos y los sesgos, de manera eficiente para que aprenda y no desaprenda,valor numerico que corresponde a la taza de aprendizaje si es un numero muy pequeño aprendera muy lento si el numero es muy grande se pasara de lo esperado
    mean_squared_error: es la funcion que revisa errores que indica que una poco cantidad de errores gandres es peor que muchos errores pequeños
    """
);

print("Comenzando entrenamiento...");
historial = modelo.fit(celsius, farenheit, epochs=1000, verbose=False); #entrenamiento de la red 1000: corresponde a las veces que repetira el entrenamiento
print("Modelo entrenado!");
 
plt.xlabel("# Epoca");
plt.ylabel("Magnitud de perdida");
plt.plot(historial.history["loss"]); #grafica de la perdida

print("Hagamos una prediccion!");
resultado = modelo.predict([100.0]); #prediccion
print("El resultado es " + str(resultado) + " farenheit!");

print("Variables internas del modelo")
print(capa.get_weights()); #pesos y sesgos de la capa