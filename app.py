# app.py
# ------
# Responsabilidad unica: interfaz grafica con Streamlit.
# Toda la logica del modelo (carga, tokenizacion, inferencia, atencion)
# vive en inference.py. Este archivo solo construye la UI, recoge inputs
# del usuario y muestra los resultados que inference.py devuelve.
#
# Estructura de la interfaz:
#   Sidebar  : selector de variante del modelo y parametros de generacion.
#   Tab 1    : demo principal — ingresar texto y generar resumen.
#   Tab 2    : visualizacion del heatmap de atencion cruzada.
#   Tab 3    : explicacion de la arquitectura T5 y sus innovaciones.

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from inference import T5Model, AVAILABLE_MODELS, EXAMPLE_TEXTS


# ---------------------------------------------------------------------------
# Configuracion de la pagina
# ---------------------------------------------------------------------------
# Debe ser la primera llamada a Streamlit en el script; cualquier otra
# instruccion st.* antes de esta genera un error de ejecucion.
st.set_page_config(
    page_title="T5 — Resumen Automatico de Texto",
    page_icon="T5",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Carga del modelo con cache de Streamlit
# ---------------------------------------------------------------------------
# @st.cache_resource almacena el objeto T5Model en memoria entre reruns.
# Streamlit re-ejecuta el script completo en cada interaccion del usuario;
# sin este decorador el modelo se volveria a descargar y cargar cada vez.
# El cache se invalida unicamente si cambia el argumento model_id.
@st.cache_resource(show_spinner="Cargando modelo T5 desde HuggingFace...")
def load_model(model_id: str) -> T5Model:
    # Instancia T5Model, que descarga pesos y construye el grafo en memoria.
    # La primera carga puede tomar entre 5 s (t5-small en CPU) y varios minutos
    # (t5-base en CPU); las siguientes son instantaneas gracias al cache.
    return T5Model(model_id)


# ---------------------------------------------------------------------------
# Sidebar: configuracion del experimento
# ---------------------------------------------------------------------------
# El sidebar es persistente entre tabs; los valores aqui definidos se usan
# en las tres pestanas del cuerpo principal.
with st.sidebar:
    st.title("Configuracion")

    # Selector de variante del modelo.
    # Cada opcion en AVAILABLE_MODELS mapea un nombre legible al ID de
    # HuggingFace Hub. t5-small es el defecto porque ofrece el mejor
    # balance calidad/velocidad en CPU sin necesidad de GPU.
    model_key = st.selectbox(
        "Variante del modelo",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Modelos mas grandes generan mejores resumenes pero son mas lentos.",
    )
    model_id = AVAILABLE_MODELS[model_key]
    # Mostrar el ID exacto de HuggingFace Hub para que el usuario pueda
    # buscarlo y verificar la arquitectura y los pesos utilizados.
    st.caption(f"`{model_id}`")

    st.divider()

    # Parametros de generacion que se pasan directamente a model.generate().
    # Se exponen en sliders para permitir experimentos sin modificar el codigo.
    st.subheader("Parametros de generacion")

    max_length = st.slider(
        "Longitud maxima del resumen (tokens)",
        min_value=50, max_value=300, value=150, step=10,
        help="Un token equivale aproximadamente a 0.75 palabras en ingles.",
    )
    min_length = st.slider(
        "Longitud minima del resumen (tokens)",
        min_value=10, max_value=100, value=30, step=5,
        help="El decoder no detendra la generacion antes de este limite.",
    )
    num_beams = st.slider(
        "Numero de beams (beam search)",
        min_value=1, max_value=8, value=4, step=1,
        help="1 = greedy decoding (rapido). Mayor numero = mejor calidad pero mas lento.",
    )
    length_penalty = st.slider(
        "Penalizacion de longitud",
        min_value=0.5, max_value=3.0, value=2.0, step=0.5,
        help=">1.0 favorece resumenes mas largos. <1.0 favorece resumenes mas cortos.",
    )

    st.divider()
    st.caption("Universidad Autonoma de Occidente · 2025")
    st.caption("Procesamiento de Datos Secuenciales con Deep Learning")


# ---------------------------------------------------------------------------
# Encabezado del cuerpo principal
# ---------------------------------------------------------------------------
st.title("T5 — Resumen Automatico de Texto")
st.markdown(
    "Implementacion del articulo **Exploring the Limits of Transfer Learning "
    "with a Unified Text-to-Text Transformer** · Google Research · 2019"
)
st.divider()

# Definicion de las tres pestanas principales.
# Streamlit ejecuta el contenido de cada bloque with tab_* en el orden
# en que aparece en el script; las tres pestanas siempre se renderizan,
# aunque el usuario solo vea la activa.
tab_demo, tab_attention, tab_architecture = st.tabs(
    ["Demo: Resumir texto", "Atencion Cruzada (Q · K · V)", "Arquitectura T5"]
)


# ===========================================================================
# PESTAÑA 1: Demo principal — generacion de resumen
# ===========================================================================
with tab_demo:
    st.subheader("Ingresa un texto en ingles para resumir")

    # Selector de textos de ejemplo predefinidos.
    # Permite cargar rapidamente textos de dominio conocido para demostrar
    # el funcionamiento sin necesidad de copiar texto manualmente.
    example_choice = st.selectbox(
        "Cargar texto de ejemplo",
        options=["(ingresar manualmente)"] + list(EXAMPLE_TEXTS.keys()),
    )

    # Si el usuario selecciona un ejemplo, se pre-rellena el area de texto.
    # Si selecciona "(ingresar manualmente)", EXAMPLE_TEXTS.get devuelve ""
    # y el area aparece vacia con el placeholder.
    default_text = EXAMPLE_TEXTS.get(example_choice, "")
    input_text = st.text_area(
        "Texto de entrada",
        value=default_text,
        height=220,
        placeholder="Pega aqui el texto que deseas resumir...",
        help="El modelo agrega automaticamente el prefijo 'summarize:' antes del texto.",
    )

    # Layout de dos columnas: boton a la izquierda, nota informativa a la derecha.
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_button = st.button("Resumir", type="primary", use_container_width=True)
    with col_info:
        st.info(
            "**Framework text-to-text:** T5 recibe `summarize: [texto]` y genera el resumen. "
            "El mismo modelo puede ejecutar otras tareas cambiando unicamente el prefijo.",
        )

    # Logica de inferencia: se ejecuta solo cuando el usuario presiona el boton.
    if run_button:
        if not input_text.strip():
            st.warning("Por favor ingresa un texto antes de resumir.")
        else:
            # load_model esta cacheado; esta llamada es O(1) si el modelo_id
            # no cambio desde la ultima vez que se cargo.
            model = load_model(model_id)

            with st.spinner("Generando resumen..."):
                result = model.summarize(
                    text=input_text,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                )

            # Mostrar el resumen con estilo visual destacado.
            st.success("**Resumen generado:**")
            st.markdown(
                f"<div style='background:#f0f7ff;padding:16px;border-radius:8px;"
                f"border-left:4px solid #1976D2;font-size:16px'>{result['summary']}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("")

            # Metricas de inferencia en cuatro columnas.
            # compression_ratio = tokens_entrada / tokens_salida; mayor ratio
            # indica que el modelo condenso mas informacion en menos tokens.
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tokens de entrada",    result["input_tokens"])
            m2.metric("Tokens de salida",     result["output_tokens"])
            m3.metric("Ratio de compresion",  f"{result['compression_ratio']}x")
            m4.metric("Tiempo de inferencia", f"{result['elapsed_seconds']} s")

            # Persistir el par (entrada, resumen) en session_state para que
            # la pestana de atencion cruzada pueda acceder a estos valores
            # sin necesidad de volver a ejecutar el modelo.
            st.session_state["last_input"]   = input_text
            st.session_state["last_summary"] = result["summary"]
            st.session_state["model_id"]     = model_id


# ===========================================================================
# PESTAÑA 2: Visualizacion de atencion cruzada
# ===========================================================================
with tab_attention:
    st.subheader("Visualizacion de la Atencion Cruzada")
    st.markdown(
        """
        La **atencion cruzada** (*cross-attention*) es el mecanismo por el cual el decoder
        decide que partes del texto de entrada son relevantes al generar cada token del resumen.

        | Tensor | Origen | Significado |
        |--------|--------|-------------|
        | **Q** (Query)  | Decoder | "Que informacion necesito para generar este token?" |
        | **K** (Key)    | Encoder | "Que informacion tengo disponible en la entrada?" |
        | **V** (Value)  | Encoder | "El contenido real que se extrae de la entrada" |

        El calculo es: **Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V**
        """
    )
    st.divider()

    # Verificar que el usuario haya generado un resumen previamente.
    # session_state persiste entre reruns del script dentro de la misma sesion.
    if "last_input" not in st.session_state:
        st.info("Primero genera un resumen en la pestana **Demo** para visualizar la atencion.")
    else:
        st.write(f"**Entrada:** {st.session_state['last_input'][:120]}...")
        st.write(f"**Resumen:** {st.session_state['last_summary']}")

        if st.button("Visualizar atencion cruzada", type="primary"):
            model = load_model(st.session_state["model_id"])

            with st.spinner("Calculando pesos de atencion..."):
                attn_data = model.get_cross_attention(
                    input_text=st.session_state["last_input"],
                    output_text=st.session_state["last_summary"],
                    max_enc_tokens=60,
                    max_dec_tokens=30,
                )

            attn   = attn_data["attention"]
            x_labs = attn_data["encoder_tokens"]
            y_labs = attn_data["decoder_tokens"]

            # Construir el heatmap con seaborn sobre un eje de matplotlib.
            # El tamano de la figura se escala dinamicamente al numero de tokens
            # para mantener las celdas legibles independientemente de la longitud
            # del texto. Los limites (20, 12) evitan figuras demasiado grandes.
            fig, ax = plt.subplots(
                figsize=(
                    min(len(x_labs) * 0.45 + 2, 20),
                    min(len(y_labs) * 0.55 + 2, 12),
                )
            )
            sns.heatmap(
                attn,
                xticklabels=x_labs,
                yticklabels=y_labs,
                cmap="Blues",          # colores mas oscuros = mayor atencion
                ax=ax,
                linewidths=0.3,
                linecolor="white",
                cbar_kws={"label": "Peso de atencion"},
            )
            ax.set_xlabel("Tokens de entrada (Encoder)", fontsize=11)
            ax.set_ylabel("Tokens de salida (Decoder)", fontsize=11)
            ax.set_title(
                "Atencion Cruzada — Ultima capa del Decoder (promedio de cabezas)",
                fontsize=12, pad=12,
            )
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)

            st.caption(
                "Cada celda muestra cuanta atencion pone el decoder en un token del encoder "
                "al generar el token de salida correspondiente. "
                "Colores mas oscuros = mayor atencion."
            )


# ===========================================================================
# PESTAÑA 3: Arquitectura T5
# ===========================================================================
with tab_architecture:
    st.subheader("Arquitectura T5 — Encoder-Decoder Transformer")

    # Descripcion del encoder y el decoder en columnas paralelas para facilitar
    # la comparacion directa de sus diferencias estructurales.
    col_enc, col_dec = st.columns(2)

    with col_enc:
        st.markdown("### Encoder")
        st.markdown(
            """
            Procesa el texto de entrada **bidireccionalmente**.
            Cada token puede atender a todos los demas tokens de la secuencia.

            **Capas por bloque:**
            1. **Multi-Head Self-Attention**
               - Q, K, V provienen de la misma secuencia
               - Captura dependencias dentro del texto de entrada
            2. **Feed-Forward Network** (2 capas lineales + activacion)
            3. **Layer Normalization** (pre-norm en T5)
            4. **Residual connections**

            **T5 usa Relative Position Bias** en lugar del encoding
            posicional sinusoidal del Transformer original.
            """
        )

    with col_dec:
        st.markdown("### Decoder")
        st.markdown(
            """
            Genera el texto de salida **autoregresivamente**,
            un token a la vez.

            **Capas por bloque:**
            1. **Masked Self-Attention**
               - Q, K, V provienen de los tokens ya generados
               - La mascara evita "ver el futuro"
            2. **Cross-Attention** — el mecanismo central
               - **Q** proviene del decoder
               - **K y V** provienen del encoder
               - Aqui el decoder "lee" el texto de entrada
            3. **Feed-Forward Network**
            4. **Layer Normalization + Residual**
            """
        )

    st.divider()
    st.markdown("### Innovaciones de T5 frente al Transformer original")

    # Cada innovacion se presenta en un expander colapsable para no saturar
    # la vista inicial y permitir al usuario profundizar solo en lo que le interesa.
    innovations = {
        "Framework Text-to-Text": (
            "Toda tarea NLP se formula como seq2seq. Un solo modelo para "
            "traduccion, resumen, clasificacion, Q&A — solo cambia el prefijo de entrada."
        ),
        "Relative Position Bias": (
            "En lugar de sumar un encoding posicional absoluto a los embeddings, "
            "T5 anade un sesgo aprendido a los logits de atencion basado en la "
            "distancia relativa entre tokens. Generaliza mejor a secuencias largas."
        ),
        "Pre-entrenamiento en C4": (
            "Entrenado con span corruption sobre C4 (750 GB de texto limpio de la web). "
            "Se corrompen spans aleatorios del texto y el modelo aprende a reconstruirlos."
        ),
        "Sin sesgos en capas densas": (
            "A diferencia del Transformer original, T5 elimina los terminos bias "
            "en las proyecciones lineales Q, K, V, reduciendo parametros sin perder calidad."
        ),
        "Variantes Efficient-T5": (
            "Familia de modelos con configuraciones optimizadas de d_model, capas y cabezas "
            "para distintos balances entre rendimiento y costo computacional."
        ),
    }

    for name, description in innovations.items():
        with st.expander(name):
            st.write(description)

    st.divider()
    st.markdown("### Limitaciones de T5")
    st.markdown(
        """
        - **Idioma:** Los modelos base estan entrenados principalmente en ingles (corpus C4).
        - **Alucinaciones:** Como todo LLM, puede generar texto plausible pero factualmente incorrecto.
        - **Costo computacional:** El modelo base requiere GPU para inferencia en tiempo real.
        - **Longitud de contexto:** Limitado a 512 tokens de entrada en la configuracion estandar.
        - **Sin memoria:** Cada inferencia es independiente; no recuerda ejecuciones anteriores.
        - **Sesgo del corpus:** C4 refleja sesgos presentes en la web en ingles.
        """
    )
