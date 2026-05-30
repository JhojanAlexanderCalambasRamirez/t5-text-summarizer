# app.py
# ------
# Responsabilidad única: interfaz gráfica con Streamlit.
# Toda la lógica del modelo (carga, tokenización, inferencia, ROUGE,
# benchmarking y atención) vive en inference.py. Este archivo solo construye
# la UI, recoge inputs del usuario y muestra resultados.
#
# Mejoras incluidas:
#   1. Selector de Layer y Head para cross-attention.
#   2. Evaluación ROUGE-1 / ROUGE-2 / ROUGE-L.
#   3. Comparación Tiny / Small / Base / Efficient.
#   4. Interpretación automática del heatmap.

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from inference import AVAILABLE_MODELS, EXAMPLE_TEXTS, MODEL_METADATA, T5Model


# ---------------------------------------------------------------------------
# Configuración de la pagina
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="T5 — Resumen Automatico de Texto",
    page_icon="T5",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Carga del modelo con cache de Streamlit
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo T5 desde HuggingFace...")
def load_model(model_id: str) -> T5Model:
    return T5Model(model_id)


# ---------------------------------------------------------------------------
# Utilidades de UI
# ---------------------------------------------------------------------------
def render_summary_box(summary: str) -> None:
    st.markdown(
        f"""
        <div style="
            background:#f0f7ff;
            color:black;
            padding:16px;
            border-radius:8px;
            border-left:4px solid #1976D2;
            font-size:16px;
            line-height:1.6;
        ">
            {summary}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_download_payload(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Sidebar: configuración del experimento
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Configuración")

    model_key = st.selectbox(
        "Variante del modelo",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Modelos mas grandes generan mejores resumenes pero son mas lentos.",
    )
    model_id = AVAILABLE_MODELS[model_key]
    st.caption(f"`{model_id}`")

    st.divider()

    st.subheader("Parámetros de generación")

    max_length = st.slider(
        "Longitud máxima del resumen (tokens)",
        min_value=50,
        max_value=300,
        value=150,
        step=10,
        help="Un token equivale aproximadamente a 0.75 palabras en inglés.",
    )
    min_length = st.slider(
        "Longitud mínima del resumen (tokens)",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="El decoder no detendrá la generación antes de este límite.",
    )
    num_beams = st.slider(
        "Número de beams (beam search)",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help="1 = greedy decoding. Mayor número = mejor calidad pero más lento.",
    )
    length_penalty = st.slider(
        "Penalización de longitud",
        min_value=0.5,
        max_value=3.0,
        value=2.0,
        step=0.5,
        help=">1.0 favorece resúmenes más largos. <1.0 favorece resúmenes más cortos.",
    )

    st.divider()
    st.caption("Universidad Autónoma de Occidente · 2025")
    st.caption("Procesamiento de Datos Secuenciales con Deep Learning")


# ---------------------------------------------------------------------------
# Encabezado del cuerpo principal
# ---------------------------------------------------------------------------
st.title("T5 — Resumen Automático de Texto")
st.markdown(
    "Implementación del artículo **Exploring the Limits of Transfer Learning "
    "with a Unified Text-to-Text Transformer** · Google Research · 2019"
)
st.divider()

(
    tab_demo,
    tab_attention,
    tab_rouge,
    tab_benchmark,
    tab_architecture,
) = st.tabs(
    [
        "Demo: Resumir texto",
        "Atención Cruzada: Layer / Head",
        "ROUGE",
        "Comparación de modelos",
        "Arquitectura T5",
    ]
)


# ==========================================================================
# PESTAÑA 1: Demo principal — generación de resumen
# ==========================================================================
with tab_demo:
    st.subheader("Ingresa un texto en inglés para resumir")

    example_choice = st.selectbox(
        "Cargar texto de ejemplo",
        options=["(ingresar manualmente)"] + list(EXAMPLE_TEXTS.keys()),
    )

    default_text = EXAMPLE_TEXTS.get(example_choice, "")
    input_text = st.text_area(
        "Texto de entrada",
        value=default_text,
        height=220,
        placeholder="Pega aquí el texto que deseas resumir...",
        help="El modelo agrega automáticamente el prefijo 'summarize:' antes del texto.",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_button = st.button("Resumir", type="primary", use_container_width=True)
    with col_info:
        st.info(
            "**Framework text-to-text:** T5 recibe `summarize: [texto]` y genera el resumen. "
            "El mismo modelo puede ejecutar otras tareas cambiando únicamente el prefijo."
        )

    if run_button:
        if not input_text.strip():
            st.warning("Por favor ingresa un texto antes de resumir.")
        else:
            model = load_model(model_id)

            with st.spinner("Generando resumen..."):
                result = model.summarize(
                    text=input_text,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                )

            st.success("**Resumen generado:**")
            render_summary_box(result["summary"])

            st.markdown("")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Tokens de entrada", result["input_tokens"])
            m2.metric("Tokens de salida", result["output_tokens"])
            m3.metric("Ratio de compresion", f"{result['compression_ratio']}x")
            m4.metric("Tiempo", f"{result['elapsed_seconds']} s")
            m5.metric("Dispositivo", result["device"])

            if "cuda_memory_allocated_mb" in result:
                g1, g2 = st.columns(2)
                g1.metric("VRAM usada", f"{result['cuda_memory_allocated_mb']} MB")
                g2.metric("VRAM reservada", f"{result['cuda_memory_reserved_mb']} MB")

            st.session_state["last_input"] = input_text
            st.session_state["last_summary"] = result["summary"]
            st.session_state["last_result"] = result
            st.session_state["model_id"] = model_id

            st.download_button(
                "Descargar resultado JSON",
                data=build_download_payload(result),
                file_name="t5_summary_result.json",
                mime="application/json",
            )


# ==========================================================================
# PESTAÑA 2: Visualización de atención cruzada con Layer / Head
# ==========================================================================
with tab_attention:
    st.subheader("Visualización avanzada de la Atención Cruzada")
    st.markdown(
        """
        La **cross-attention** es el mecanismo por el cual el decoder consulta la salida
        del encoder mientras genera el resumen. Ahora puedes escoger la **capa del decoder**
        y la **cabeza de atención** especifica.

        - **Layer**: profundidad del decoder desde la capa inicial hasta la ultima.
        - **Head**: una cabeza individual dentro de Multi-Head Attention.
        - **Promedio**: combina varias cabezas o capas para una vista global.
        """
    )
    st.divider()

    if "last_input" not in st.session_state:
        st.info("Primero genera un resumen en la pestana **Demo** para visualizar la atención.")
    else:
        model = load_model(st.session_state["model_id"])
        model_info = model.get_model_info()

        num_decoder_layers = int(model_info.get("num_decoder_layers") or model_info.get("num_layers") or 6)
        num_heads_model = int(model_info.get("num_heads") or 8)

        st.write(f"**Entrada:** {st.session_state['last_input'][:160]}...")
        st.write(f"**Resumen:** {st.session_state['last_summary']}")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            layer_option = st.selectbox(
                "Decoder Layer",
                options=["Promedio de capas"] + [f"Layer {i}" for i in range(num_decoder_layers)],
                index=num_decoder_layers,
                help="La ultima capa suele reflejar decisiones mas cercanas a la generacion final.",
            )

        with c2:
            head_option = st.selectbox(
                "Attention Head",
                options=["Promedio de heads"] + [f"Head {i}" for i in range(num_heads_model)],
                index=0,
                help="Cada cabeza puede aprender patrones distintos de alineamiento.",
            )

        with c3:
            max_enc_tokens = st.slider(
                "Tokens encoder",
                min_value=20,
                max_value=120,
                value=60,
                step=10,
            )

        with c4:
            max_dec_tokens = st.slider(
                "Tokens decoder",
                min_value=10,
                max_value=80,
                value=30,
                step=10,
            )

        average_layers = layer_option == "Promedio de capas"
        layer_idx = None if average_layers else int(layer_option.split()[-1])
        head_idx = None if head_option == "Promedio de heads" else int(head_option.split()[-1])

        if st.button("Visualizar atención cruzada", type="primary"):
            with st.spinner("Calculando pesos de atención..."):
                attn_data = model.get_cross_attention(
                    input_text=st.session_state["last_input"],
                    output_text=st.session_state["last_summary"],
                    max_enc_tokens=max_enc_tokens,
                    max_dec_tokens=max_dec_tokens,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    average_layers=average_layers,
                )

            st.session_state["last_attention_data"] = attn_data

        if "last_attention_data" in st.session_state:
            attn_data = st.session_state["last_attention_data"]
            attn = attn_data["attention"]
            x_labs = attn_data["encoder_tokens"]
            y_labs = attn_data["decoder_tokens"]

            title = f"Atención Cruzada — {attn_data['selected_layer']} — {attn_data['selected_head']}"

            fig, ax = plt.subplots(
                figsize=(
                    min(len(x_labs) * 0.45 + 2, 22),
                    min(len(y_labs) * 0.55 + 2, 14),
                )
            )
            sns.heatmap(
                attn,
                xticklabels=x_labs,
                yticklabels=y_labs,
                cmap="Blues",
                ax=ax,
                linewidths=0.3,
                linecolor="white",
                cbar_kws={"label": "Peso de atención"},
            )
            ax.set_xlabel("Tokens de entrada (Encoder)", fontsize=11)
            ax.set_ylabel("Tokens de salida (Decoder)", fontsize=11)
            ax.set_title(title, fontsize=12, pad=12)
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)

            st.caption(
                "Cada celda indica cuánta atención pone el decoder en un token del encoder "
                "al generar el token de salida correspondiente. Colores más oscuros = mayor atención."
            )

            st.markdown("### Interpretación automática")
            interpretation = attn_data["interpretation"]
            st.info(interpretation["message"])

            top_tokens = interpretation["top_tokens"]
            if top_tokens:
                df_tokens = pd.DataFrame(top_tokens)
                st.dataframe(
                    df_tokens[["index", "clean_token", "token", "score"]],
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "Descargar datos de atención JSON",
                data=build_download_payload(
                    {
                        "selected_layer": attn_data["selected_layer"],
                        "selected_head": attn_data["selected_head"],
                        "encoder_tokens": attn_data["encoder_tokens"],
                        "decoder_tokens": attn_data["decoder_tokens"],
                        "interpretation": attn_data["interpretation"],
                    }
                ),
                file_name="t5_cross_attention_data.json",
                mime="application/json",
            )


# ==========================================================================
# PESTAÑA 3: ROUGE
# ==========================================================================
with tab_rouge:
    st.subheader("Evaluación con ROUGE")
    st.markdown(
        """
        ROUGE compara el resumen generado contra un **resumen de referencia**.
        Es una metrica común en tareas de resumen automático.

        - **ROUGE-1**: coincidencia de unigramas.
        - **ROUGE-2**: coincidencia de bigramas.
        - **ROUGE-L**: subsequence común mas larga.
        """
    )

    if "last_summary" not in st.session_state:
        st.info("Primero genera un resumen en la pestana **Demo**.")
    else:
        st.markdown("**Resumen generado por el modelo:**")
        render_summary_box(st.session_state["last_summary"])

        reference_summary = st.text_area(
            "Resumen de referencia / esperado",
            height=160,
            placeholder="Pega aquí un resumen humano o esperado para compararlo con el generado por T5.",
        )

        if st.button("Calcular ROUGE", type="primary"):
            if not reference_summary.strip():
                st.warning("Debes ingresar un resumen de referencia para calcular ROUGE.")
            else:
                try:
                    scores = T5Model.compute_rouge(
                        reference_text=reference_summary,
                        generated_text=st.session_state["last_summary"],
                    )
                    st.session_state["last_rouge"] = scores
                except ImportError as exc:
                    st.error(str(exc))

        if "last_rouge" in st.session_state:
            scores = st.session_state["last_rouge"]
            rows = []
            for metric, values in scores.items():
                rows.append(
                    {
                        "Metrica": metric.upper(),
                        "Precision": values["precision"],
                        "Recall": values["recall"],
                        "F1": values["fmeasure"],
                    }
                )
            df_scores = pd.DataFrame(rows)
            st.dataframe(df_scores, use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("ROUGE-1 F1", scores["rouge1"]["fmeasure"])
            c2.metric("ROUGE-2 F1", scores["rouge2"]["fmeasure"])
            c3.metric("ROUGE-L F1", scores["rougeL"]["fmeasure"])


# ==========================================================================
# PESTAÑA 4: Comparación de modelos
# ==========================================================================
with tab_benchmark:
    st.subheader("Comparación de modelos T5")
    st.markdown(
        """
        Esta sección ejecuta el mismo texto con varios modelos T5 y compara:

        - Tiempo de inferencia.
        - Tiempo por token generado.
        - Longitud del resumen.
        - Ratio de compresión.
        - Numero de parámetros.

        Esto permite analizar el compromiso entre calidad, velocidad y costo computacional.
        """
    )

    benchmark_text = st.text_area(
        "Texto para comparar modelos",
        value=st.session_state.get(
            "last_input",
            EXAMPLE_TEXTS["Redes Neuronales"],
        ),
        height=200,
    )

    default_benchmark_models = [
        "t5-small",
        "t5-base",
    ]

    benchmark_model_keys = st.multiselect(
        "Modelos a comparar",
        options=list(AVAILABLE_MODELS.keys()),
        default=[
            key
            for key in default_benchmark_models
            if key in AVAILABLE_MODELS
        ],
        help="Evita seleccionar demasiados modelos si estas trabajando en CPU.",
    )

    st.info(
        "La primera ejecución puede tardar porque HuggingFace descarga los pesos "
        "del modelo y los almacena en cache."
    )

    if st.button(
        "Ejecutar comparación",
        type="primary",
        key="run_benchmark",
    ):

        if not benchmark_text.strip():
            st.warning("Ingresa un texto para comparar.")

        elif not benchmark_model_keys:
            st.warning("Selecciona al menos un modelo.")

        else:

            benchmark_rows = []
            progress = st.progress(0)

            for i, key in enumerate(benchmark_model_keys):

                current_model_id = AVAILABLE_MODELS[key]

                try:

                    with st.spinner(f"Ejecutando {key}..."):

                        bm_model = load_model(current_model_id)

                        bm_result = bm_model.summarize(
                            text=benchmark_text,
                            max_length=max_length,
                            min_length=min_length,
                            num_beams=num_beams,
                            length_penalty=length_penalty,
                        )

                        bm_info = bm_model.get_model_info()

                    benchmark_rows.append(
                        {
                            "Modelo": key,
                            "HF ID": current_model_id,
                            "Parámetros approx.": MODEL_METADATA.get(
                                key,
                                {},
                            ).get(
                                "parameters",
                                "N/D",
                            ),
                            "Parámetros reales": bm_info.get(
                                "parameter_count_total"
                            ),
                            "Dispositivo": bm_result["device"],
                            "Tiempo (s)": bm_result[
                                "elapsed_seconds"
                            ],
                            "Tiempo/token (s)": round(
                                bm_result["elapsed_seconds"]
                                /
                                max(
                                    bm_result["output_tokens"],
                                    1,
                                ),
                                4,
                            ),
                            "Tokens entrada": bm_result[
                                "input_tokens"
                            ],
                            "Tokens salida": bm_result[
                                "output_tokens"
                            ],
                            "Longitud resumen": len(
                                bm_result["summary"].split()
                            ),
                            "Compresión": bm_result[
                                "compression_ratio"
                            ],
                            "Resumen": bm_result["summary"],
                        }
                    )

                except Exception as e:

                    benchmark_rows.append(
                        {
                            "Modelo": key,
                            "HF ID": current_model_id,
                            "Error": str(e),
                        }
                    )

                progress.progress(
                    (i + 1)
                    /
                    len(benchmark_model_keys)
                )

            st.session_state[
                "benchmark_rows"
            ] = benchmark_rows

    # ------------------------------------------------------------------
    # Mostrar resultados
    # ------------------------------------------------------------------

    if "benchmark_rows" in st.session_state:

        benchmark_rows = st.session_state[
            "benchmark_rows"
        ]

        df_benchmark = pd.DataFrame(
            benchmark_rows
        )

        st.markdown(
            "### Resultados de la comparación"
        )

        st.dataframe(
            df_benchmark,
            use_container_width=True,
            hide_index=True,
        )

        # ------------------------------------------------------
        # Graficos
        # ------------------------------------------------------

        if "Tiempo (s)" in df_benchmark.columns:

            st.markdown(
                "### Tiempo de inferencia"
            )

            st.bar_chart(
                df_benchmark.set_index(
                    "Modelo"
                )[["Tiempo (s)"]]
            )

        if "Tiempo/token (s)" in df_benchmark.columns:

            st.markdown(
                "### Tiempo por token"
            )

            st.bar_chart(
                df_benchmark.set_index(
                    "Modelo"
                )[["Tiempo/token (s)"]]
            )

        if "Compresión" in df_benchmark.columns:

            st.markdown(
                "### Ratio de compresión"
            )

            st.bar_chart(
                df_benchmark.set_index(
                    "Modelo"
                )[["Compresión"]]
            )

        if "Longitud resumen" in df_benchmark.columns:

            st.markdown(
                "### Longitud del resumen"
            )

            st.bar_chart(
                df_benchmark.set_index(
                    "Modelo"
                )[["Longitud resumen"]]
            )

        # ------------------------------------------------------
        # Resumenes
        # ------------------------------------------------------

        st.markdown(
            "### Resúmenes generados"
        )

        for row in benchmark_rows:

            if "Resumen" not in row:
                continue

            with st.expander(
                f"{row['Modelo']} — {row['Tiempo (s)']} s"
            ):
                st.write(row["Resumen"])

        # ------------------------------------------------------
        # Exportar
        # ------------------------------------------------------

        st.download_button(
            "Descargar comparación JSON",
            data=build_download_payload(
                {
                    "benchmark": benchmark_rows
                }
            ),
            file_name="t5_model_benchmark.json",
            mime="application/json",
        )  

# ==========================================================================
# PESTAÑA 5: Arquitectura T5
# ==========================================================================
with tab_architecture:
    st.subheader("Arquitectura T5 — Encoder-Decoder Transformer")

    model_for_info = load_model(model_id)
    info = model_for_info.get_model_info()

    st.markdown("### Configuración real del modelo cargado")
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric("d_model", info.get("d_model"))
    i2.metric("d_ff", info.get("d_ff"))
    i3.metric("Heads", info.get("num_heads"))
    i4.metric("Layers", info.get("num_layers"))
    i5.metric("Vocab", info.get("vocab_size"))

    st.caption(
        f"Modelo: `{info['model_id']}` · Dispositivo: `{info['device']}` · "
        f"Parámetros reales: {info['parameter_count_total']:,}"
    )

    st.divider()

    col_enc, col_dec = st.columns(2)

    with col_enc:
        st.markdown("### Encoder")
        st.markdown(
            """
            Procesa el texto de entrada **bidireccionalmente**.
            Cada token puede atender a todos los demás tokens de la secuencia.

            **Capas por bloque:**
            1. **Multi-Head Self-Attention**
               - Q, K, V provienen de la misma secuencia.
               - Captura dependencias dentro del texto de entrada.
            2. **Feed-Forward Network**.
            3. **Layer Normalization** pre-norm.
            4. **Residual connections**.
            """
        )

    with col_dec:
        st.markdown("### Decoder")
        st.markdown(
            """
            Genera el texto de salida **autoregresivamente**, un token a la vez.

            **Capas por bloque:**
            1. **Masked Self-Attention**.
            2. **Cross-Attention**.
               - **Q** proviene del decoder.
               - **K y V** provienen del encoder.
               - Aqui el decoder lee el texto de entrada.
            3. **Feed-Forward Network**.
            4. **Layer Normalization + Residual**.
            """
        )

    st.divider()
    st.markdown("### Innovaciones de T5 frente al Transformer original")

    innovations = {
        "Framework Text-to-Text": (
            "Toda tarea NLP se formula como texto a texto. Un solo modelo puede servir "
            "para traducción, resumen, clasificación o preguntas y respuestas cambiando el prefijo."
        ),
        "Relative Position Bias": (
            "T5 añade un sesgo aprendido a los logits de atención basado en distancia relativa "
            "entre tokens, en vez de sumar posiciones absolutas a los embeddings."
        ),
        "Span Corruption": (
            "Durante el preentrenamiento se reemplazan grupos de tokens por tokens centinela "
            "y el decoder aprende a reconstruir esos fragmentos."
        ),
        "Pre-Norm": (
            "La normalización ocurre antes de cada subcapa, lo que estabiliza el entrenamiento "
            "de redes profundas."
        ),
        "Visualización por Layer y Head": (
            "El proyecto permite inspeccionar capas y cabezas especificas de cross-attention, "
            "en lugar de mostrar solo el promedio global."
        ),
    }

    for name, description in innovations.items():
        with st.expander(name):
            st.write(description)

    st.divider()
    st.markdown("### Limitaciones de T5")
    st.markdown(
        """
        - **Idioma:** Los modelos base están entrenados principalmente en ingles.
        - **Alucinaciones:** Puede generar texto plausible pero factualmente incorrecto.
        - **Costo computacional:** Las variantes grandes son lentas en CPU.
        - **Longitud de contexto:** La configuración estándar se limita a 512 tokens de entrada.
        - **Sin memoria:** Cada inferencia es independiente.
        - **Sesgo del corpus:** C4 refleja sesgos presentes en la web en ingles.
        """
    )

