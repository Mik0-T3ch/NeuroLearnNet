<H1>¿Que son redes neuronales?</H1>

<H3>Primero debemos entender qué es una neurona en el contexto de Machine Learning.
En esta área, la neurona básica recibe el nombre de perceptrón el perceptrón es un modelo matemático simple que recibe varias entradas, las combina mediante pesos, y produce una salida aplicando una función de activación. Este modelo es la base para el entrenamiento de sistemas más complejos, como modelos de Machine Learning y Deep Learning a través del entrenamiento con datasets, el perceptrón puede ajustar sus pesos para reducir errores, aprender patrones y mejorar su desempeño de forma iterativa.</H3>

![Redes Neuronales de Grafos](https://images.squarespace-cdn.com/content/v1/5d0c74548c8e8f0001be73ba/1625093768220-DXWZPDITT3RA2Q77VPOF/Rredes+neuronales+grafos.gif?format=2500w)

<br>

<H1>El perceptron esta inspirado en una neurona biologica</H1>

<H3>La neurona biológica es la unidad fundamental del sistema nervioso y su función principal es recibir, procesar e interpretar información para generar una respuesta. Este proceso ocurre mediante señales electroquímicas que permiten la comunicación entre neuronas y otras células del cuerpo.
<br>
  <br>
Las señales entrantes son recibidas a través de las dendritas, las cuales transmiten los estímulos hacia el soma o cuerpo celular. En el soma, estas señales se integran y se evalúa su efecto conjunto. Si la suma de los estímulos recibidos es suficiente para superar un cierto umbral eléctrico, la neurona se activa y genera un potencial de acción.
<br>
  <br>
El potencial de acción se propaga a lo largo del axón sin perder intensidad hasta llegar a las terminales sinápticas, donde la señal es transmitida a otras neuronas mediante la liberación de neurotransmisores. Dependiendo del tipo de neurotransmisor y del receptor, esta señal puede tener un efecto excitatorio o inhibitorio en la neurona receptora.
<br>
  <br>
Este mecanismo de integración de múltiples señales, evaluación mediante un umbral y generación de una salida es la base conceptual que inspira la neurona artificial. En modelos computacionales, este comportamiento se abstrae mediante entradas, pesos y funciones de activación, permitiendo replicar de forma simplificada el funcionamiento de una neurona biológica.</H3>

<img width="710" height="285" alt="image" src="https://github.com/user-attachments/assets/d08d9449-ad16-41ba-b3c7-bbf7f9dd2768" />

<H1>¿Que es un perceptron?</H1>

<H3>Un perceptrón es el modelo más simple de una neurona artificial y
representa la unidad básica sobre la que se construyen las redes neuronales.
<br>
<br>
Este modelo recibe múltiples entradas, las combina mediante pesos y calcula
una suma ponderada que es procesada por una función de activación para generar
una salida. Matemáticamente, el perceptrón implementa un clasificador lineal.
<br>
<br>
Durante el entrenamiento, el perceptrón ajusta sus pesos de forma iterativa
con el objetivo de reducir el error entre la salida predicha y la salida
esperada. Sin embargo, debido a su estructura simple, solo puede resolver
problemas que son linealmente separables.<H3>
<br>
<img width="675" height="320" alt="image" src="https://github.com/user-attachments/assets/96689f40-2d0a-495f-9613-92e4df353dca" />

<H1>Arquitectura del perceptrón monocapa</H1>


<p style="font-size: 16px; line-height: 1.6;"> <h3>
  El <b>perceptrón monocapa</b> es la forma más simple de una red neuronal artificial.
  Su arquitectura está compuesta por <b>una sola neurona</b> (sin capas ocultas) y se utiliza principalmente
  para problemas de <b>clasificación binaria</b> que sean <b>linealmente separables</b>. </h3>
</p>

<hr>

<h3 style="font-size: 22px; margin-top: 18px;"> Entradas (Inputs)</h3>
<p style="font-size: 16px; line-height: 1.6;">
 <h4> Las entradas representan las características o variables del problema. Se suelen representar como un vector: </h4>
</p>

<pre style="font-size: 15px; padding: 12px; border-radius: 8px; background: #111; color: #eee; overflow-x: auto;">
x = (x1, x2, ..., xn)
</pre>

<p style="font-size: 16px; line-height: 1.6;">
 <h4> Cada valor <b>xi</b> es un número que influye directamente en la salida final del perceptrón. </h4>
</p>

<hr>

<h3 style="font-size: 22px; margin-top: 18px;"> Pesos (Weights)</h3>
<p style="font-size: 16px; line-height: 1.6;">
<h4> Cada entrada tiene asociado un peso que representa su importancia: </h4> 
</p>

<pre style="font-size: 15px; padding: 12px; border-radius: 8px; background: #111; color: #eee; overflow-x: auto;">
w = (w1, w2, ..., wn)
</pre>

<p style="font-size: 16px; line-height: 1.6;">
 <h4> Durante el entrenamiento, estos pesos se ajustan para reducir el error entre la predicción y la salida esperada. </h4>
</p>

<hr>

<h3 style="font-size: 22px; margin-top: 18px;"> Sesgo (Bias)</h3>
<p style="font-size: 16px; line-height: 1.6;">
<h4>  El sesgo (<b>b</b>) es un valor adicional que permite desplazar el umbral de decisión del modelo, haciendo que
  el perceptrón sea más flexible: </h4>
</p>

<pre style="font-size: 15px; padding: 12px; border-radius: 8px; background: #111; color: #eee; overflow-x: auto;">
b = valor constante
</pre>

<hr>

<h3 style="font-size: 22px; margin-top: 18px;"> Suma ponderada</h3>
<p style="font-size: 16px; line-height: 1.6;">
 <h4>El perceptrón calcula una combinación lineal entre entradas y pesos, sumando también el sesgo: </h4>
</p>

<pre style="font-size: 15px; padding: 12px; border-radius: 8px; background: #111; color: #eee; overflow-x: auto;">
z = (x1*w1 + x2*w2 + ... + xn*wn) + b
</pre>

<hr>

<h3 style="font-size: 22px; margin-top: 18px;"> Función de activación</h3>
<p style="font-size: 16px; line-height: 1.6;">
 <h4> Luego, el valor <b>z</b> pasa por una función de activación (comúnmente <b>escalón</b> o <b>signo</b>)
  para producir una salida: </h4>
</p>

<pre style="font-size: 15px; padding: 12px; border-radius: 8px; background: #111; color: #eee; overflow-x: auto;">
y = 1   si z >= 0
y = 0   si z < 0
</pre>

<hr>

<h3 style="font-size: 22px; margin-top: 18px;"> Salida (Output)</h3>
<p style="font-size: 16px; line-height: 1.6;">
<h4> La salida final es un valor binario (<b>0</b> o <b>1</b>), lo que hace que el perceptrón sea útil para
  problemas de clasificación simple.</h4> 
</p>

<p style="font-size: 16px; line-height: 1.6;">

<img width="1224" height="503" alt="image" src="https://github.com/user-attachments/assets/0f31292f-14c4-438a-90ed-269c8b5fded7" />

