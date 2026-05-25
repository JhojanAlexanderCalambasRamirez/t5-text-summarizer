# T5 — Resumen Automático de Texto con Transformer Encoder-Decoder

**Procesamiento de Datos Secuenciales con Deep Learning**  
Universidad Autónoma de Occidente · 2025

| Integrante | Código |
|---|---|
| Alexander Calambas Ramirez | 22602907 |
| Angelo Parra Cortez | 22506988 |
| Oscar Portela Ospina | 22507314 |
| Sebastian Torres Cabrera | 22507322 |

---

## 1. Resumen (Abstract)

Este proyecto implementa inferencia sobre el modelo **T5 (Text-to-Text Transfer Transformer)** de Google Research aplicado a la tarea de resumen automático de texto en inglés. T5 es una arquitectura Transformer encoder-decoder que unifica todas las tareas de procesamiento de lenguaje natural bajo un único framework: dado un texto de entrada con un prefijo de tarea, el modelo genera el texto de salida correspondiente. Se utilizan los pesos preentrenados disponibles en HuggingFace (`t5-small`, `t5-base` y variantes efficient) sin necesidad de entrenar desde cero. La interfaz interactiva desarrollada con Streamlit permite ingresar cualquier texto, visualizar el resumen generado, consultar métricas de compresión y explorar los pesos de atención cruzada entre encoder y decoder mediante un mapa de calor. Los resultados demuestran que un solo modelo preentrenado puede resumir textos de distintos dominios con ratios de compresión entre 3x y 8x manteniendo coherencia semántica.

---

## 2. Introducción

### Artículo base

**Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer."** *Journal of Machine Learning Research*, 21(140), 1-67.

- Artículo en HuggingFace Papers: [https://huggingface.co/papers/1910.10683](https://huggingface.co/papers/1910.10683)
- Repositorio original (Google Research): [https://github.com/google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)
- Pesos preentrenados utilizados: [google/t5-efficient-small](https://huggingface.co/google/t5-efficient-small) y las variantes del modelo T5 original en [T5 community](https://huggingface.co/google-t5)

### Contexto y problemática

Antes de T5, el estado del arte en procesamiento de lenguaje natural requería un modelo especializado para cada tarea: uno para traducción automática, otro para resumen de texto, otro para clasificación de sentimientos, otro para respuesta a preguntas. Este enfoque tenía tres problemas fundamentales:

1. **Fragmentación:** cada tarea requería arquitectura, datos y proceso de entrenamiento propios.
2. **Ineficiencia:** los aprendizajes de una tarea no se transferían a otras.
3. **Escalabilidad:** añadir una nueva tarea implicaba construir un sistema completamente nuevo.

### Solución propuesta por T5

T5 propone que **todas las tareas de NLP pueden reformularse como un problema de texto a texto**. El modelo recibe una cadena de texto con un prefijo que identifica la tarea y produce otra cadena de texto como salida:

```
Entrada:  "summarize: [texto largo]"         → Salida: "[resumen]"
Entrada:  "translate English to French: [x]" → Salida: "[traducción]"
Entrada:  "cola sentence: [oración]"         → Salida: "acceptable" o "unacceptable"
```

Un único modelo preentrenado resuelve todas estas tareas de forma unificada.

### Objetivo del proyecto

Implementar inferencia funcional con T5 para resumen automático de texto, desarrollar una interfaz interactiva que permita explorar la arquitectura y los mecanismos de atención, y demostrar el concepto text-to-text durante la sustentación.

---

## 3. Marco teórico

### 3.1 Arquitectura Transformer Encoder-Decoder original y la variante T5

La arquitectura Transformer convencional, introducida por Vaswani et al. [2], supuso una ruptura con los modelos basados en redes neuronales recurrentes y convolucionales al fundamentar el procesamiento de secuencias exclusivamente en mecanismos de atención. 

El bloque del encoder transforma una secuencia de representaciones continuas de tokens de entrada $X = (x_1, ..., x_n)$ en una secuencia de vectores ocultos o contextualizados $Z = (z_1, ..., z_n)$. Cada capa del encoder consta de dos subcapas principales: un mecanismo de atención multi-cabeza autorregresivo bidireccional y una red neuronal Position-Wise Feed-Forward. 

Por otra parte, el bloque del decoder genera una secuencia de salida $Y = (y_1, ..., y_m)$ de forma autorregresiva, es decir, token a token, utilizando las representaciones contextuales $Z$ provistas por el encoder y los tokens previamente generados. El decoder clásico añade una tercera subcapa intermedia dedicada a la atención cruzada (cross-attention), la cual conecta funcionalmente ambos bloques.

![encoder-decoder.png](screenshots/encoder-decoder.png)

El modelo Text-to-Text Transfer Transformer (T5), formulado por Raffel et al. [1], adopta esta estructura secuencial clásica, pero introduce modificaciones estructurales fundamentales para optimizar la eficiencia y la estabilidad del gradiente durante el aprendizaje por transferencia a gran escala. A diferencia de las tendencias contemporáneas que simplificaron la arquitectura hacia configuraciones de solo encoder como BERT o solo decoder como GPT, T5 sostiene que mantener la arquitectura encoder-decoder resulta óptimo para resolver tareas generales de secuencias complejas de texto a texto.

### 3.2 Mecanismo Multi-Head Attention

El núcleo operacional de la arquitectura Transformer es la **Scaled Dot-Product Attention**. Este mecanismo computa la relevancia mutua entre los elementos de las secuencias mapeando matrices de consultas ($Q$), claves ($K$) y valores ($V$). Las proyecciones lineales se obtienen a partir de una matriz de entrada de activaciones $X$ multiplicada por matrices de pesos entrenables, definidas formalmente de la siguiente manera.

La matriz de consultas se expresa como:
$$Q = XW_Q$$

La matriz de claves se expresa como:
$$K = XW_K$$

La matriz de valores se expresa como:
$$V = XW_V$$

Donde las dimensiones de los pesos corresponden a $W_Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_K \in \mathbb{R}^{d_{model} \times d_k}$ y $W_V \in \mathbb{R}^{d_{model} \times d_v}$. La función matemática que rige la asignación de pesos de atención y la agregación del contexto se define mediante la ecuación de producto escalar escalado:

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

El factor de escala $\frac{1}{\sqrt{d_k}}$ mitiga el crecimiento desproporcionado de las magnitudes de los productos escalares cuando la dimensionalidad $d_k$ es elevada, evitando que la función softmax sature en regiones de gradiente infinitesimalmente pequeño.

![scaled_dot-product_attention_multi-head_attention.png](screenshots/scaled_dot-product_attention_multi-head_attention.png)

Para enriquecer la capacidad de representación, se implementa la Multi-Head Attention. En lugar de realizar una única operación de atención sobre las dimensiones globales, el modelo proyecta linealmente $h$ veces las consultas, claves y valores de forma independiente en subespacios dimensionales reducidos. Cada proyección se procesa en paralelo mediante la función de atención escalada, y las salidas resultantes se concatenan para ser proyectadas nuevamente a la dimensión original del modelo:

$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W_O$$

Donde cada cabeza individual se calcula como:
$$head_i = Attention(QW_{Q,i}, KW_{K,i}, VW_{V,i})$$

Los parámetros de proyección por cada cabeza corresponden a las matrices de pesos $W_{Q,i} \in \mathbb{R}^{d_{model} \times d_k}$, $W_{K,i} \in \mathbb{R}^{d_{model} \times d_k}$, $W_{V,i} \in \mathbb{R}^{d_{model} \times d_v}$ y la proyección de salida $W_O \in \mathbb{R}^{h d_v \times d_{model}}$. Esta descomposición permite que el sistema atienda simultáneamente a información proveniente de diferentes subespacios de representación y distintas coordenadas posicionales.

### 3.3 Tipos de atención en la arquitectura T5

La variante T5 distribuye el mecanismo de Multi-Head Attention en tres modalidades funcionales diferenciadas a lo largo de su estructura secuencial, controlando de forma estricta el flujo de información y las dependencias temporales.

| Tipo de atención                      | Dónde                                 | $Q$                                                                               | $K$                                                                                                         | $V$                                                                                                         | Propósito                                                                                                                                                                                                                                                                            |
|:--------------------------------------|:--------------------------------------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Self-Attention del Encoder**        | Capas internas del bloque Encoder     | Derivada de las activaciones de la capa previa del encoder                        | Derivada de las activaciones de la capa previa del encoder                                                  | Derivada de las activaciones de la capa previa del encoder                                                  | Permite una codificación completamente bidireccional, donde cada token del texto de entrada atiende a todos los demás tokens de la secuencia sin restricciones de causalidad.                                                                                                        |
| **Masked Self-Attention del Decoder** | Capas inferiores del bloque Decoder   | Derivada de las activaciones de la capa previa del decoder                        | Derivada de las activaciones de la capa previa del decoder                                                  | Derivada de las activaciones de la capa previa del decoder                                                  | Limita el campo receptivo mediante una máscara causal que inicializa con valor de infinito negativo los elementos superiores de la matriz de atención, previniendo que el decoder acceda a información de tokens futuros durante el entrenamiento autoregresivo.                     |
| **Cross-Attention**                   | Subcapa intermedia del bloque Decoder | Proviene directamente de la normalización de la subcapa previa dentro del decoder | Proviene directamente de las representaciones contextuales finales generadas por la última capa del encoder | Proviene directamente de las representaciones contextuales finales generadas por la última capa del encoder | Actúa como el puente de transferencia de información. El decoder emite consultas para buscar correspondencias semánticas dentro de las claves y valores del encoder, extrayendo el conocimiento sintáctico de la secuencia original necesario para guiar la generación de la salida. |

### 3.4 Innovaciones arquitectónicas de T5

El desarrollo de T5 introdujo variaciones respecto al diseño de Vaswani de 2017 [2], redefiniendo los estándares de estabilidad en el entrenamiento y la flexibilidad en la transferencia de conocimiento de los modelos masivos de lenguaje.

#### 3.4.1 Framework Text-to-Text unificado 

La innovación principal propuesta por Raffel et al. [1] consiste en la unificación conceptual de todas las tareas del procesamiento de lenguaje natural bajo un único formato secuencial de entrada y salida de texto. Al anteponer un prefijo descriptivo explícito (por ejemplo, el prefijo de tarea para este proyecto consiste en la cadena de texto de entrada traducida a tokens como un comando explícito `summarize: `), el modelo reutiliza exactamente la misma arquitectura, la función de pérdida por entropía cruzada y la estrategia de decodificación para tareas estructuralmente diferentes como la traducción, la clasificación, la regresión y el resumen automático.

#### 3.4.2 Sesgo de posición relativa o Relative Position Bias 

T5 prescinde por completo de los embeddings de posición absoluta de naturaleza sinusoidal o aprendida aplicados directamente sobre la entrada del modelo. En su lugar, implementa un sesgo posicional relativo donde los logits de atención se modifican en función de la distancia matemática existente entre el token de consulta y el token de clave. La función de atención modificada adopta la siguiente formulación matemática:

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V$$

Donde $B$ representa una matriz de sesgo posicional entrenable. Para optimizar la eficiencia computational, T5 emplea una asignación por compartimentos logarítmicos (logarithmic bucketing), la cual asigna un parámetro de sesgo único para distancias relativas cortas y agrupa distancias mayores en un número fijo de categorías. Esta innovación otorga una capacidad de generalización superior al procesar secuencias significativamente más extensas que aquellas presentes durante el régimen de preentrenamiento.

#### 3.4.3 Preentrenamiento mediante corrupción de fragmentos o Span Corruption en C4 

En contraposición al enmascaramiento de tokens unitarios implementado en arquitecturas tradicionales como BERT, T5 fundamenta su preentrenamiento auto-supervisado en la tarea de corrupción de fragmentos continuos de texto, reemplazándolos con sentinel tokens especiales. Este proceso se ejecutó sobre el conjunto de datos Colossal Clean Crawled Corpus o C4, un corpus de aproximadamente 750 GB de texto web filtrado mediante reglas heurísticas estrictas para remover contenido repetitivo o sintácticamente defectuoso. Esta aproximación entrena de forma nativa la naturaleza generativa del bloque decoder para la reconstrucción secuencial de secuencias lingüísticas coherentes.

![span-corruption.png](screenshots/span-corruption.png)

#### 3.4.4 Arquitectura de linealidad sin sesgo o Bias-Free Dense Layers 

Con el propósito de optimizar el consumo de memoria en hardware de aceleración masiva y suprimir parámetros redundantes, T5 elimina por completo los vectores de sesgo aditivo en todas las transformaciones lineales asociadas a las proyecciones densas de consultas, claves y valores, así como en las capas internas de la Feed-Forward Network. Las operaciones correspondientes conservan estrictamente una naturaleza multiplicativa de matrices de pesos ponderados.

#### 3.4.5 Estabilización por RMSNorm en configuración Pre-Norm 

T5 modifica la topología de las conexiones residuales del Transformer clásico. En lugar de aplicar la normalización de capa de manera posterior a la suma residual, sitúa el bloque de normalización de forma previa a la ejecución de cada subcapa, manteniendo un canal libre para la propagación del gradiente en arquitecturas de gran profundidad. Adicionalmente, sustituye la técnica LayerNorm convencional por RMSNorm (Root Mean Square Normalization) formulada por Zhang y Sennrich [7]. Esta variante prescinde de la operación de centrado basada en la media y limita el cómputo exclusivamente al escalado por la raíz de la media de los cuadrados, lo que reduce el costo computational por iteración en el orden del 7-10% sin detrimento de la convergencia de la red.

---

## 4. Metodología

### 4.1 Proceso de implementación y criterios de selección

El desarrollo del presente proyecto se estructuró en tres etapas: el análisis de las restricciones de cómputo locales, el diseño de un canal de inferencia desacoplado para el procesamiento de texto y la construcción de un entorno visual interactivo que permitiera auditar las decisiones del modelo de forma gráfica.

Debido a que el despliegue y la validación del sistema se ejecutan en hardware de consumo general sin acceso a unidades de procesamiento gráfico dedicadas, se realizó un análisis comparativo de la viabilidad de cómputo en CPU para las distintas variantes de T5. Aunque los modelos de parámetros escalados como `t5-base` brindan una precisión semántica elevada, su exigencia computacional por cada paso autoregresivo del decoder introduce latencias que comprometen la interactividad en tiempo real de la interfaz. 

Bajo estas condiciones, se optó por la variante `google/t5-efficient-small`. Esta arquitectura modifica la profundidad y el número de cabezas de atención respecto al diseño estándar, ofreciendo un balance óptimo al reducir la dimensionalidad de las representaciones internas sin degradar críticamente la coherencia gramatical del resumen generado. Con aproximadamente 60 millones de parámetros, esta variante permite sostener tiempos de inferencia en CPU inferiores a los 4 segundos por secuencia.

### 4.2 Herramientas utilizadas

| Herramienta              | Versión | Propósito en el proyecto                                                                                   |
|:-------------------------|:--------|:-----------------------------------------------------------------------------------------------------------|
| Python                   | 3.10+   | Entorno de ejecución e interpretación del código base.                                                     |
| PyTorch                  | >=2.0   | Backend de cómputo numérico, gestión de tensores y ejecución del grafo de inferencia.                      |
| HuggingFace Transformers | >=4.40  | API de abstracción para la inicialización de la arquitectura T5 y la inyección de pesos preentrenados.     |
| SentencePiece            | >=0.2   | Motor de tokenización basado en subpalabras independiente del idioma para codificación de texto.           |
| Streamlit                | >=1.35  | Framework para el desarrollo acelerado de la interfaz gráfica de usuario y renderizado de componentes web. |
| Matplotlib / Seaborn     | >=3.7   | Generación y formateo de matrices bidimensionales para la visualización de la atención cruzada.            |

### 4.3 Interfaz interactiva en Streamlit

Para democratizar el acceso al modelo y permitir la auditoría de los mecanismos internos de atención, se diseñó una interfaz gráfica interactiva dividida funcionalmente en dos capas de abstracción mediante componentes de pestañas. 

La primera capa se enfoca en el control operativo del modelo, exponiendo controles en la barra lateral para parametrizar hiperparámetros del pipeline generativo, tales como la longitud mínima y máxima de los tokens de salida, y el uso de penalizaciones por repetición. La segunda capa está destinada al análisis explicativo del modelo de aprendizaje profundo, proporcionando un espacio dedicado a la representación visual de las matrices de peso interceptadas.

### 4.4 Extracción de los pesos de atención cruzada

La evaluación profunda de la arquitectura exige capturar los pesos calculados por la función softmax en la subcapa de atención cruzada del decoder. En condiciones normales de inferencia, las matrices intermedias de alineación no se retienen en memoria para optimizar el uso de recursos. Para subvertir esta restricción sin alterar el flujo computacional de PyTorch.

Matemáticamente, para la última capa del decoder, la matriz de atención cruzada que mapea la influencia de los tokens generados sobre los tokens de entrada se extrae directamente de la tupla de tensores devuelta por el modelo. Dado que la atención multi-cabeza procesa $h$ cabezas en paralelo, el tensor extraído posee una forma cuatridimensional indexada por:

$$\mathcal{A} \in \mathbb{R}^{\text{batch_size} \times h \times \text{target_seq_len} \times \text{source_seq_len}}$$

Para efectos de visualización en este proyecto, se extrae el lote correspondiente a la inferencia actual, y se calcula el promedio aritmético o la selección selectiva a través de las $h$ cabezas de la última capa del decoder, reduciendo el tensor a una matriz bidimensional apta para el mapeo térmico:

$$M_{j,i} = \frac{1}{h} \sum_{c=1}^{h} \mathcal{A}_{c, j, i}$$

Donde $j$ indexa las posiciones de los tokens de salida generados por el resumen y $i$ indexa las posiciones de los tokens de la secuencia original codificada por el encoder.

### 4.5 Uso de pesos preentrenados

El proyecto no realiza ningún proceso de optimización de pesos ni entrenamiento desde cero. Se consumen los parámetros preentrenados de la distribución de eficiencia de Google disponibles en HuggingFace Hub. El proceso de inicialización y almacenamiento en las estructuras de datos de PyTorch se gestiona mediante el siguiente bloque de código.

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-small")
model = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-small")
```

Durante la primera ejecución del script, los archivos de parametrización binaria se descargan de forma automatizada y se localizan en el directorio de persistencia local del sistema operativo (~/.cache/huggingface/), permitiendo que las ejecuciones subsecuentes prescindan de conectividad a la red de internet.

### 4.6 Variantes del modelo disponibles

| Modelo             | Parámetros    | Entorno recomendado                                                                                                     |
|:-------------------|:--------------|:------------------------------------------------------------------------------------------------------------------------|
| t5-efficient-tiny  | ~6 millones   | Pruebas de integración rápidas entornos con severas restricciones de memoria RAM o CPUs antiguas.                       |
| t5-efficient-mini  | ~11 millones  | Ejecución estándar en CPU con latencias de procesamiento moderadas.                                                     |
| t5-efficient-small | ~60 millones  | Configuración seleccionada en este proyecto; balance óptimo entre preservación semántica y velocidad de cómputo en CPU. |
| t5-efficient-base  | ~250 millones | Despliegue mandatorio en hardware con aceleración de GPU; máxima fidelidad interpretativa.                              |

---

## 5. Desarrollo e Implementación

### Estructura del proyecto

```
T5/
├── README.md          ← documentación completa (este archivo)
├── inference.py       ← lógica T5: carga, generación, atención
├── app.py             ← interfaz Streamlit
├── requirements.txt   ← dependencias
└── screenshots/       ← capturas de pantalla propias
```

**Principios de diseño:** alta cohesión (cada archivo tiene una sola responsabilidad) y bajo acoplamiento (`app.py` solo conoce la interfaz pública de `inference.py`).

### Cómo ejecutar el proyecto

**1. Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd T5
```

**2. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**3. Ejecutar la aplicación**
```bash
streamlit run app.py
```

La primera ejecución descarga los pesos (~250 MB para t5-efficient-small). Las siguientes usan la caché local.

### Flujo de preprocesamiento e inferencia

```
1. Usuario ingresa texto en la interfaz
        ↓
2. Se añade el prefijo de tarea:
   "summarize: " + texto_usuario
        ↓
3. Tokenización SentencePiece:
   texto → secuencia de IDs de tokens
   (max 512 tokens, truncación si es necesario)
        ↓
4. Encoder forward pass:
   IDs → embeddings → N bloques de atención
   → representaciones contextuales H
        ↓
5. Decoder autoregresivo (beam search, N beams):
   Para cada paso t:
   a. Masked self-attention sobre tokens ya generados
   b. Cross-attention: Q=decoder, K=H, V=H
   c. FFN → distribución sobre vocabulario
   d. Selección del token con mayor probabilidad
   e. Repetir hasta </s> o max_length
        ↓
6. Decodificación:
   IDs de salida → texto del resumen
        ↓
7. Visualización en Streamlit
```

### Extracción de pesos de atención

Para visualizar la atención cruzada se realiza un forward pass adicional con `output_attentions=True`:

```python
outputs = model(
    input_ids=encoder_input,
    decoder_input_ids=decoder_input,
    output_attentions=True,
    return_dict=True,
)
# outputs.cross_attentions: tupla con tensor por capa
# Forma: (batch, num_heads, dec_seq_len, enc_seq_len)
cross_attn_last = outputs.cross_attentions[-1][0]  # última capa
attn_avg = cross_attn_last.mean(dim=0)             # promedio de cabezas
```

---

## 6. Resultados y Análisis

### Ejemplo de inferencia — Inteligencia Artificial

**Entrada (180 tokens):**
> *"Artificial intelligence (AI) is intelligence demonstrated by machines... AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars..."*

**Salida — Resumen generado por T5:**
> *"artificial intelligence is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals. AI applications include advanced web search engines, recommendation systems, and self-driving cars."*

| Métrica | Valor |
|---------|-------|
| Tokens de entrada | ~180 |
| Tokens de salida | ~40 |
| Ratio de compresión | ~4.5x |
| Tiempo de inferencia (CPU) | ~2-4 s |
| Modelo utilizado | t5-efficient-small |

### Análisis de la atención cruzada

El mapa de calor de atención cruzada revela el comportamiento del mecanismo Q-K-V durante la generación:

- Los tokens del resumen que capturan ideas principales muestran **atención concentrada** en los tokens de entrada correspondientes (colores oscuros en el heatmap).
- Los tokens funcionales (`the`, `a`, `is`) distribuyen la atención de forma más uniforme.
- Se observa que el modelo asigna mayor atención a sustantivos y verbos clave del texto fuente, ignorando palabras de relleno.

**Capturas de pantalla de la aplicación:**

![Demo inicial](screenshots/01_demo_inicial.png)
*Tab 1 — Demo: interfaz de entrada con selector de ejemplos y parámetros en sidebar.*

![Mapa de atención cruzada](screenshots/02_attention_heatmap.png)
*Tab 2 — Heatmap de atención cruzada: el decoder "lee" el encoder token a token.*

### Comparación de variantes

| Modelo | Calidad del resumen | Velocidad (CPU) | Tamaño |
|--------|--------------------|-----------------|----|
| t5-small  | Buena, coherente ✓ | Moderada (~0.5s) | ~240 MB |
| t5-base   | Muy buena | Lenta (~2-4s) | ~900 MB |
| efficient-tiny | Básica, omite detalles | Muy rápida | ~25 MB |
| efficient-small | Aceptable | Rápida | ~60 MB |
| efficient-base  | Muy buena | Moderada-lenta | ~250 MB |

---

## 7. Conclusiones

### Aprendizajes

- El framework text-to-text de T5 demuestra que la unificación de tareas NLP no solo es posible sino que mejora el rendimiento general al compartir representaciones entre tareas.
- La **atención cruzada** es el mecanismo que hace funcionar la conexión encoder-decoder: el decoder literalmente "consulta" el texto de entrada en cada paso de generación, y esto es observable y visualizable directamente en los pesos Q-K-V.
- El **Relative Position Bias** es una innovación que mejora significativamente la generalización a secuencias largas comparado con el encoding posicional absoluto.
- Los pesos preentrenados de HuggingFace permiten implementar inferencia de calidad sin recursos de entrenamiento, lo que democratiza el acceso a modelos de lenguaje avanzados.

### Limitaciones

- **Idioma:** El modelo base está optimizado para inglés. Para otros idiomas se requieren variantes multilingües (mT5).
- **Alucinaciones:** T5 puede generar texto gramaticalmente correcto pero factualmente incorrecto, especialmente con textos que contienen datos numéricos o nombres propios poco frecuentes.
- **Longitud máxima de contexto:** 512 tokens de entrada. Textos más largos se truncan, perdiendo información del final.
- **Costo computacional:** El modelo `t5-efficient-base` requiere GPU para inferencia en tiempo real en producción.
- **Sin memoria conversacional:** Cada inferencia es completamente independiente.

### Posibles mejoras

- Fine-tuning sobre el dataset CNN/DailyMail para mejorar la calidad del resumen.
- Integrar mT5 para soporte de español y otros idiomas.
- Ampliar la interfaz para soportar múltiples tareas (traducción, Q&A, clasificación) demostrando el concepto text-to-text en toda su extensión.
- Implementar visualización de la atención de encoder (self-attention) además de la cross-attention.

---

## 8. Referencias

[1] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer," *Journal of Machine Learning Research*, vol. 21, no. 140, pp. 1–67, 2020.

[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[3] HuggingFace, "google/t5-efficient-small," HuggingFace Hub. [Online]. Available: https://huggingface.co/google/t5-efficient-small

[4] HuggingFace, "google/t5-efficient-base," HuggingFace Hub. [Online]. Available: https://huggingface.co/google/t5-efficient-base

[5] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," in *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pp. 38–45, 2020.

[6] HuggingFace, "Papers with Code — T5," 2019. [Online]. Available: https://huggingface.co/papers/1910.10683

[7] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in Advances in Neural Information Processing Systems, vol. 32, 2019.
