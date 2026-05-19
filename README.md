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
- Pesos preentrenados utilizados: [google/t5-efficient-small](https://huggingface.co/google/t5-efficient-small)

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

**5. Layer Normalization pre-norm**  
T5 aplica la normalización antes de la sublayer (pre-norm) en lugar de después (post-norm), lo que estabiliza el entrenamiento de modelos muy profundos.

---

## 4. Metodología

### Herramientas utilizadas

| Herramienta | Versión | Propósito |
|-------------|---------|-----------|
| Python | 3.10+ | Lenguaje base |
| PyTorch | ≥2.0 | Backend de cómputo tensorial |
| HuggingFace Transformers | ≥4.40 | Carga de modelo y tokenizador T5 |
| SentencePiece | ≥0.2 | Tokenizador de subpalabras de T5 |
| Streamlit | ≥1.35 | Interfaz interactiva |
| Matplotlib / Seaborn | ≥3.7 | Visualización de atención |

### Uso de pesos preentrenados

El proyecto **no entrena el modelo desde cero**. Se utilizan los pesos preentrenados de `google/t5-efficient-small` disponibles públicamente en HuggingFace Hub. La descarga ocurre automáticamente la primera vez que se ejecuta la aplicación:

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-small")
model     = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-small")
```

Los pesos quedan en caché local (`~/.cache/huggingface/`) para ejecuciones posteriores sin conexión.

### Variantes del modelo disponibles

| Modelo | Parámetros aprox. | Uso recomendado |
|--------|------------------|-----------------|
| t5-efficient-tiny  | ~6M   | Pruebas rápidas, CPU lento |
| t5-efficient-mini  | ~11M  | CPU, latencia aceptable |
| t5-efficient-small | ~60M  | Balance calidad/velocidad ✓ |
| t5-efficient-base  | ~250M | GPU, mejor calidad |

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
