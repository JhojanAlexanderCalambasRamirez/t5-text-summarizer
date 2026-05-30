# T5 — Resumen Automático de Texto con Transformer Encoder-Decoder

**Procesamiento de Datos Secuenciales con Deep Learning**  
Universidad Autónoma de Occidente · 2026

| Integrante | Código |
|---|---|
| Alexander Calambas Ramirez | 22602907 |
| Angelo Parra Cortez | 22506988 |
| Oscar Portela Ospina | 22507314 |
| Sebastian Torres Cabrera | 22507322 |

**Repositorio:** [https://github.com/JhojanAlexanderCalambasRamirez/t5-text-summarizer](https://github.com/JhojanAlexanderCalambasRamirez/t5-text-summarizer)

---

## 1. Resumen (Abstract)

Este proyecto implementa inferencia sobre el modelo **T5 (Text-to-Text Transfer Transformer)** de Google Research aplicado a la tarea de resumen automático de texto en inglés. T5 es una arquitectura Transformer encoder-decoder que unifica todas las tareas de procesamiento de lenguaje natural bajo un único framework: dado un texto de entrada con un prefijo de tarea, el modelo genera el texto de salida correspondiente. Se utilizan los pesos preentrenados disponibles en HuggingFace (`t5-small` y `t5-base`) sin necesidad de entrenar desde cero. La interfaz interactiva desarrollada con Streamlit ofrece cinco pestañas: demostración de resumen, visualización configurable de atención cruzada por capa y cabeza, evaluación con métricas ROUGE, comparación entre variantes del modelo y explicación de la arquitectura T5. Los resultados demuestran que un solo modelo preentrenado puede resumir textos de distintos dominios con ratios de compresión entre 3x y 8x manteniendo coherencia semántica.

---

## 2. Introducción

### 2.1 Artículo base

**Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer."** *Journal of Machine Learning Research*, 21(140), 1-67.

- Artículo en HuggingFace Papers: [https://huggingface.co/papers/1910.10683](https://huggingface.co/papers/1910.10683)
- Repositorio original (Google Research): [https://github.com/google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)
- Pesos preentrenados utilizados: [t5-small](https://huggingface.co/google-t5/t5-small) y [t5-base](https://huggingface.co/google-t5/t5-base)

### 2.2 Contexto y problemática

Antes de T5, el estado del arte en procesamiento de lenguaje natural requería un modelo especializado para cada tarea: uno para traducción automática, otro para resumen de texto, otro para clasificación de sentimientos, otro para respuesta a preguntas. Este enfoque tenía tres problemas fundamentales:

1. **Fragmentación:** cada tarea requería arquitectura, datos y proceso de entrenamiento propios.
2. **Ineficiencia:** los aprendizajes de una tarea no se transferían a otras.
3. **Escalabilidad:** añadir una nueva tarea implicaba construir un sistema completamente nuevo.

### 2.3 Solución propuesta por T5

T5 propone que **todas las tareas de NLP pueden reformularse como un problema de texto a texto**. El modelo recibe una cadena de texto con un prefijo que identifica la tarea y produce otra cadena de texto como salida:

```
Entrada:  "summarize: [texto largo]"         → Salida: "[resumen]"
Entrada:  "translate English to French: [x]" → Salida: "[traducción]"
Entrada:  "cola sentence: [oración]"         → Salida: "acceptable" o "unacceptable"
```

Un único modelo preentrenado resuelve todas estas tareas de forma unificada.

### 2.4 Objetivo del proyecto

Implementar inferencia funcional con T5 para resumen automático de texto, desarrollar una interfaz interactiva que permita explorar la arquitectura y los mecanismos de atención, evaluar la calidad del resumen con métricas ROUGE y demostrar el concepto text-to-text durante la sustentación.

---

## 3. Marco Teórico

### 3.1 Arquitectura Transformer Encoder-Decoder

T5 se basa en la arquitectura Transformer original propuesta por Vaswani et al. (2017), en su variante encoder-decoder. La arquitectura procesa secuencias de tokens en dos etapas:

```
Texto de entrada
       ↓
  [Tokenización]
       ↓
   ENCODER  ──────────────────────────────────────────────
   │  Bloque × N                                         │
   │  ├── Multi-Head Self-Attention                      │
   │  │    Q = XW_Q,  K = XW_K,  V = XW_V               │
   │  │    (X = tokens de entrada)                       │
   │  ├── Feed-Forward Network                           │
   │  └── Layer Norm + Residual                          │
   └──── Representaciones contextuales H                 │
                                                         │ Cross-Attention
   DECODER  ◄────────────────────────────────────────────┘
   │  Bloque × N                                         
   │  ├── Masked Self-Attention                          
   │  │    (tokens ya generados, no puede ver el futuro) 
   │  ├── Cross-Attention  ← MECANISMO CLAVE             
   │  │    Q = decoder,  K = H,  V = H                   
   │  ├── Feed-Forward Network                           
   │  └── Layer Norm + Residual                          
   └──── Distribución sobre vocabulario → token generado 
```

### 3.2 Mecanismo de Atención Multi-Cabeza

El corazón del Transformer es la atención por producto escalado (*Scaled Dot-Product Attention*):

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

**¿Qué es cada tensor?**

| Tensor | Generación | Función |
|--------|-----------|---------|
| **Q** (Query) | `Q = X · W_Q` — proyección lineal de la secuencia query | "¿Qué información estoy buscando?" |
| **K** (Key)   | `K = X · W_K` — proyección lineal de la secuencia key   | "¿Qué información puedo ofrecer?" |
| **V** (Value) | `V = X · W_V` — proyección lineal de la secuencia value | "El contenido real que entrego" |

El término `√d_k` normaliza los productos punto para evitar gradientes demasiado pequeños cuando la dimensionalidad `d_k` es grande (problema de saturación del softmax).

**Multi-Head Attention:** en lugar de aplicar una sola función de atención, T5 aplica `h` funciones de atención en paralelo sobre subespacios proyectados de menor dimensión y concatena los resultados:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
donde head_i = Attention(Q · W_Q_i, K · W_K_i, V · W_V_i)
```

Esto permite al modelo atender simultáneamente a distintos aspectos del texto desde diferentes posiciones.

### 3.3 Tipos de Atención en T5

| Tipo | Dónde ocurre | Q | K | V |
|------|-------------|---|---|---|
| **Self-Attention del Encoder** | Dentro del encoder | Tokens de entrada | Tokens de entrada | Tokens de entrada |
| **Masked Self-Attention del Decoder** | Dentro del decoder | Tokens generados | Tokens generados | Tokens generados |
| **Cross-Attention** | Decoder → Encoder | Decoder (query) | Encoder output | Encoder output |

La **Cross-Attention** es el mecanismo que conecta encoder y decoder: el decoder "pregunta" al encoder qué partes del texto original son relevantes para generar cada token de salida.

### 3.4 Innovaciones de T5

**1. Framework Text-to-Text unificado**  
La contribución principal del artículo: reformular todas las tareas NLP como seq2seq usando prefijos de tarea. Esto permite preentrenar un único modelo y adaptarlo a cualquier tarea cambiando solo el prefijo de entrada, sin modificar la arquitectura.

**2. Relative Position Bias**  
El Transformer original suma un encoding posicional sinusoidal fijo a los embeddings. T5 introduce un sesgo aprendido que se añade directamente a los logits de atención (`QKᵀ`) en función de la distancia relativa entre tokens:

```
Atención modificada = softmax( (QKᵀ + b(i-j)) / √d_k ) · V
```

donde `b(i-j)` es un escalar aprendido que depende del desplazamiento relativo entre posiciones. Esto mejora la generalización a secuencias más largas que las vistas durante el entrenamiento.

**3. Pre-entrenamiento con Span Corruption sobre C4**  
T5 se preentrenó en el corpus C4 (Colossal Clean Crawled Corpus, 750 GB de texto web limpiado) usando un objetivo de corrupción de spans: se enmascaran spans aleatorios del texto (no tokens individuales como en BERT) y el modelo aprende a reconstruirlos.

**4. Sin bias en capas densas**  
Las matrices de proyección Q, K, V y la FFN no tienen términos de bias, reduciendo el número de parámetros sin degradar el rendimiento.

**5. Layer Normalization pre-norm (RMSNorm)**  
T5 aplica RMSNorm antes de cada sublayer (pre-norm) en lugar de después (post-norm), lo que estabiliza el entrenamiento de modelos muy profundos.

---

## 4. Metodología

### 4.1 Proceso de implementación y criterios de selección

El proyecto no realiza ningún proceso de optimización de pesos ni entrenamiento desde cero. Se consumen los parámetros preentrenados disponibles en HuggingFace Hub. Se optó por `t5-small` como variante por defecto: con ~60M de parámetros produce resúmenes coherentes con tiempos de inferencia en CPU inferiores a 1 segundo. La variante `t5-base` está disponible para mayor fidelidad semántica a costa de mayor latencia.

### 4.2 Herramientas utilizadas

| Herramienta | Versión | Propósito |
|-------------|---------|-----------|
| Python | 3.10+ | Lenguaje base |
| PyTorch | ≥2.0 | Backend de cómputo tensorial e inferencia |
| HuggingFace Transformers | ≥4.40 | Carga de arquitectura T5 y pesos preentrenados |
| SentencePiece | ≥0.2 | Tokenizador de subpalabras de T5 |
| Streamlit | ≥1.35 | Interfaz interactiva |
| Matplotlib / Seaborn | ≥3.7 | Visualización del heatmap de atención cruzada |
| NumPy | ≥1.24 | Manipulación de la matriz de atención (ndarray 2D) |
| Pandas | ≥2.0 | Tablas en pestañas de benchmark y ROUGE |
| rouge-score | ≥0.1.2 | Métricas ROUGE-1, ROUGE-2 y ROUGE-L (opcional) |

### 4.3 Interfaz interactiva en Streamlit

La interfaz está dividida en cinco pestañas funcionales:

| Pestaña | Función |
|---------|---------|
| **Demo: Resumir texto** | Caja de texto, botón Resumir, métricas (tokens in/out, ratio, tiempo de inferencia) |
| **Atención Cruzada: Layer / Head** | Heatmap configurable por capa (0-5) y cabeza (0-7) o promedio; interpretación automática de tokens más consultados |
| **ROUGE** | Evaluación ROUGE-1, ROUGE-2 y ROUGE-L contra un resumen de referencia provisto por el usuario |
| **Comparación de modelos** | Benchmark entre t5-small y t5-base sobre el mismo texto con tabla comparativa de métricas |
| **Arquitectura T5** | Diagramas y explicación del encoder, decoder, innovaciones y dimensiones del modelo activo |

### 4.4 Extracción configurable de pesos de atención cruzada

La evaluación profunda de la arquitectura exige capturar los pesos de la cross-attention del decoder. Para persistirlos sin alterar el flujo computacional de PyTorch se activa `output_attentions=True`. El método `get_cross_attention` de `inference.py` permite seleccionar:

- **Capa específica** (`layer_idx=N`): cualquiera de las 6 capas del decoder (0-5).
- **Cabeza específica** (`head_idx=N`): cualquiera de las 8 cabezas de atención (0-7).
- **Promedio de capas** (`average_layers=True`): promedia las 6 capas para una visión global.
- **Promedio de cabezas** (por defecto): promedia las 8 cabezas de la capa seleccionada.

```python
outputs = model(
    input_ids=encoder_input,
    decoder_input_ids=decoder_input,
    output_attentions=True,
    return_dict=True,
)
# outputs.cross_attentions: tupla, un tensor por capa del decoder
# Forma: (batch, num_heads, dec_seq_len, enc_seq_len)
cross_attn = outputs.cross_attentions[-1][0]   # última capa, primer batch
attn_avg   = cross_attn.mean(dim=0)             # promedio de cabezas → (dec_len, enc_len)
```

Adicionalmente, el método `interpret_attention` identifica automáticamente los tokens del encoder más consultados promediando los pesos por columna y filtrando signos de puntuación.

### 4.5 Evaluación con métricas ROUGE

El método estático `compute_rouge` de `inference.py` calcula ROUGE-1, ROUGE-2 y ROUGE-L comparando el resumen generado contra un resumen de referencia provisto por el usuario:

```python
scores = T5Model.compute_rouge(reference_text=referencia, generated_text=resumen)
# Retorna: {"rouge1": {"precision": ..., "recall": ..., "fmeasure": ...}, ...}
```

Requiere `pip install rouge-score`. Si no está instalado, la pestaña ROUGE muestra un mensaje de instalación sin interrumpir el resto de la aplicación.

### 4.6 Variantes del modelo disponibles

| Modelo | Parámetros | Uso recomendado |
|--------|-----------|-----------------|
| t5-small (default) | ~60M | Resúmenes coherentes en CPU, latencia <1s. Recomendado para demos educativas. |
| t5-base | ~220M | Mayor fidelidad semántica. Latencia 2-4s en CPU. |

Las variantes `t5-efficient-*` están disponibles en el catálogo de `inference.py` (comentadas) para entornos con RAM muy limitada.

---

## 5. Desarrollo e Implementación

### 5.1 Estructura del proyecto

```
t5-text-summarizer/
├── README.md             ← documentación completa (este archivo)
├── inference.py          ← lógica T5: carga, generación, ROUGE, atención
├── app.py                ← interfaz Streamlit (5 pestañas)
├── requirements.txt      ← dependencias para CPU (macOS / Windows)
├── requirements-cuda.txt ← dependencias para GPU NVIDIA (CUDA)
└── screenshots/          ← capturas de pantalla de la aplicación
```

**Principios de diseño:** alta cohesión (cada archivo tiene una sola responsabilidad) y bajo acoplamiento (`app.py` solo conoce la interfaz pública de `inference.py`).

### 5.2 Cómo ejecutar el proyecto

**1. Clonar el repositorio**
```bash
git clone https://github.com/JhojanAlexanderCalambasRamirez/t5-text-summarizer.git
cd t5-text-summarizer
```

**2. Crear e inicializar el entorno virtual**

En macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows (PowerShell):
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Instalar dependencias**

Para CPU (macOS Intel, macOS Apple Silicon, Windows):
```bash
pip install -r requirements.txt
```

Para GPU NVIDIA (CUDA):
```bash
pip install -r requirements-cuda.txt
```

**4. Ejecutar la aplicación**
```bash
streamlit run app.py
```

La primera ejecución descarga los pesos desde HuggingFace Hub (~240 MB para t5-small). Las siguientes usan la caché local en `~/.cache/huggingface/`.

### 5.3 Flujo de preprocesamiento e inferencia

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
   IDs → embeddings → 6 bloques de atención
   → representaciones contextuales K y V
        ↓
5. Decoder autoregresivo (beam search):
   Para cada paso t:
   a. Masked self-attention sobre tokens ya generados
   b. Cross-attention: Q=decoder, K=encoder, V=encoder
   c. FFN → distribución sobre vocabulario (32.128 tokens)
   d. Selección del token con mayor probabilidad acumulada
   e. Repetir hasta </s> o max_length
        ↓
6. Decodificación:
   IDs de salida → texto del resumen
        ↓
7. Visualización en Streamlit + métricas
```

### 5.4 Métodos principales de `inference.py`

| Método | Descripción |
|--------|-------------|
| `T5Model.__init__` | Detecta dispositivo (CUDA/MPS/CPU), carga tokenizador y pesos |
| `T5Model.get_model_info` | Devuelve dimensiones reales del modelo (d_model, num_heads, etc.) desde su config |
| `T5Model.summarize` | Genera el resumen: prefijo → tokenización → beam search → decodificación |
| `T5Model.compute_rouge` | Calcula ROUGE-1, ROUGE-2 y ROUGE-L contra un resumen de referencia |
| `T5Model.get_cross_attention` | Extrae la matriz de cross-attention configurable por capa, cabeza o promedio |
| `T5Model.interpret_attention` | Identifica los tokens del encoder más consultados por el decoder |

---

## 6. Resultados y Análisis

### 6.1 Ejemplo de inferencia — Inteligencia Artificial

**Entrada (201 tokens):**
> *"Artificial intelligence (AI) is intelligence demonstrated by machines... AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars..."*

**Salida — Resumen generado por T5:**
> *"artificial intelligence is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals. AI applications include advanced web search engines, recommendation systems, and self-driving cars."*

| Métrica | Valor |
|---------|-------|
| Tokens de entrada | 201 |
| Tokens de salida | 58 |
| Ratio de compresión | 3.47x |
| Tiempo de inferencia (CPU) | ~0.5 s |
| Modelo utilizado | t5-small |

### 6.2 Análisis de la atención cruzada

El heatmap de atención cruzada permite auditar visualmente el comportamiento del mecanismo Q-K-V durante la generación. La pestaña **Atención Cruzada** permite seleccionar cualquier combinación de capa (0-5) y cabeza (0-7) del decoder para observar cómo varía el foco de atención entre capas:

- Los tokens del resumen que capturan ideas principales muestran **atención concentrada** en los tokens de entrada correspondientes (colores oscuros en el heatmap).
- Los tokens funcionales (`the`, `a`, `is`) distribuyen la atención de forma más uniforme.
- Las capas superiores (4-5) tienden a mostrar patrones más específicos y semánticamente interpretables que las capas inferiores.
- La interpretación automática debajo del heatmap lista los tokens del encoder más consultados en promedio.

**Capturas de pantalla de la aplicación:**

![Demo inicial](screenshots/01_demo_inicial.png)
*Tab 1 — Demo: interfaz de entrada con selector de ejemplos y parámetros en sidebar.*

![Mapa de atención cruzada](screenshots/02_attention_heatmap.png)
*Tab 2 — Heatmap de atención cruzada con selector de capa y cabeza.*

### 6.3 Comparación de variantes

| Modelo | Calidad del resumen | Velocidad (CPU) | Tamaño en disco |
|--------|--------------------|-----------------|----|
| t5-small (default) | Buena, coherente ✓ | ~0.5s | ~240 MB |
| t5-base | Muy alta fidelidad | ~2-4s | ~900 MB |

---

## 7. Conclusiones

### 7.1 Aprendizajes

- El framework text-to-text de T5 demuestra que la unificación de tareas NLP no solo es posible sino que mejora el rendimiento general al compartir representaciones entre tareas.
- La **atención cruzada** es el mecanismo que hace funcionar la conexión encoder-decoder: el decoder literalmente "consulta" el texto de entrada en cada paso de generación. La posibilidad de inspeccionar capas y cabezas individualmente revela que distintas cabezas se especializan en distintos tipos de dependencias lingüísticas.
- El **Relative Position Bias** es una innovación que mejora significativamente la generalización a secuencias largas comparado con el encoding posicional absoluto.
- Los pesos preentrenados de HuggingFace permiten implementar inferencia de calidad sin recursos de entrenamiento, lo que democratiza el acceso a modelos de lenguaje avanzados.

### 7.2 Limitaciones

- **Idioma:** El modelo base está optimizado para inglés. Para otros idiomas se requieren variantes multilingües (mT5).
- **Alucinaciones:** T5 puede generar texto gramaticalmente correcto pero factualmente incorrecto, especialmente con textos que contienen datos numéricos o nombres propios poco frecuentes.
- **Longitud máxima de contexto:** 512 tokens de entrada. Textos más largos se truncan, perdiendo información del final.
- **Costo computacional:** La variante `t5-base` requiere GPU para inferencia en tiempo real en producción.
- **Sin memoria conversacional:** Cada inferencia es completamente independiente.

### 7.3 Posibles mejoras

- Fine-tuning sobre el dataset CNN/DailyMail para mejorar la calidad del resumen y obtener puntuaciones ROUGE más altas.
- Integrar mT5 para soporte de español y otros idiomas.
- Ampliar la interfaz para soportar múltiples tareas (traducción, Q&A, clasificación) demostrando el concepto text-to-text en toda su extensión.
- Visualizar la self-attention del encoder para observar cómo el modelo codifica las dependencias internas del texto de entrada.
- Implementar exportación del heatmap y las métricas como archivo PDF o CSV desde la interfaz.

---

## 8. Referencias

[1] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer," *Journal of Machine Learning Research*, vol. 21, no. 140, pp. 1–67, 2020.

[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[3] HuggingFace, "google-t5/t5-small," HuggingFace Hub. [Online]. Available: https://huggingface.co/google-t5/t5-small

[4] HuggingFace, "google-t5/t5-base," HuggingFace Hub. [Online]. Available: https://huggingface.co/google-t5/t5-base

[5] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," in *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pp. 38–45, 2020.

[6] HuggingFace, "Papers with Code — T5," 2019. [Online]. Available: https://huggingface.co/papers/1910.10683

[7] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in Advances in Neural Information Processing Systems, vol. 32, 2019.
