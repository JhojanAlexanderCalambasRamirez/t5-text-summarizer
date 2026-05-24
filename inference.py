# inference.py
# ------------
# Responsabilidad única: toda la lógica relacionada con el modelo T5.
# Carga de pesos desde HuggingFace Hub, tokenización con SentencePiece,
# generación de resúmenes con beam search y extracción de pesos de atención
# cruzada para visualización en la interfaz.
#
# Este módulo no importa Streamlit ni conoce la UI. Solo recibe datos,
# ejecuta el modelo y devuelve resultados en estructuras Python puras (dict).

import time
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer


# ---------------------------------------------------------------------------
# Catálogo de variantes del modelo
# ---------------------------------------------------------------------------
# t5-small / t5-base: modelos estándar preentrenados en C4 con span corruption.
#   Producen resúmenes coherentes. Recomendados para demos educativas en CPU.
#   t5-small  ~ 60M parámetros, t5-base ~ 220M parámetros.
#
# t5-efficient-*: variantes con arquitectura compacta diseñadas por Google para
#   reducir el número de parámetros manteniendo aceptable calidad de salida.
#   Útiles cuando el tiempo de descarga o la memoria RAM son limitados.
#
# Los IDs de la derecha son los identificadores exactos en HuggingFace Hub.
# Al instanciar T5Model, los pesos se descargan una sola vez y quedan en
# caché local (~/.cache/huggingface/) para usos posteriores sin conexión.
AVAILABLE_MODELS = {
    "t5-small":           "t5-small",
    "t5-base":            "t5-base",
    "t5-efficient-tiny":  "google/t5-efficient-tiny",
    "t5-efficient-small": "google/t5-efficient-small",
    "t5-efficient-base":  "google/t5-efficient-base",
}


# ---------------------------------------------------------------------------
# Textos de ejemplo para la interfaz de demostración
# ---------------------------------------------------------------------------
# Todos los textos están en inglés porque T5 fue preentrenado en C4, un corpus
# de páginas web en inglés de 750 GB. Sin fine-tuning específico, la calidad
# del resumen en otros idiomas no está garantizada.
EXAMPLE_TEXTS = {
    "Inteligencia Artificial": (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to "
        "the natural intelligence displayed by animals including humans. AI research has been "
        "defined as the field of study of intelligent agents, which refers to any system that "
        "perceives its environment and takes actions that maximize its chance of achieving its "
        "goals. The term 'artificial intelligence' had previously been used to describe machines "
        "that mimic and display human cognitive skills associated with the human mind, such as "
        "learning and problem-solving. This definition has since been rejected by major AI "
        "researchers who now describe AI in terms of rationality and acting rationally, which "
        "does not limit how intelligence can be articulated. AI applications include advanced "
        "web search engines, recommendation systems, understanding human speech, self-driving "
        "cars, generative tools and competing at the highest level in strategic games. As "
        "machines become increasingly capable, tasks considered to require intelligence are "
        "often removed from the definition of AI, a phenomenon known as the AI effect."
    ),
    "Cambio Climatico": (
        "Climate change refers to long-term shifts in temperatures and weather patterns. "
        "These shifts may be natural, such as through variations in the solar cycle. But "
        "since the 1800s, human activities have been the main driver of climate change, "
        "primarily due to burning fossil fuels like coal, oil and gas. Burning fossil fuels "
        "generates greenhouse gas emissions that act like a blanket wrapped around the Earth, "
        "trapping the sun's heat and raising temperatures. Examples of greenhouse gas emissions "
        "that are causing climate change include carbon dioxide and methane. These come from "
        "using gasoline for driving a car or coal for heating a building, for example. Clearing "
        "land and forests can also release carbon dioxide. Landfills for garbage are a major "
        "source of methane emissions. Energy, industry, transport, buildings, agriculture and "
        "land use are among the main emitters. The consequences of climate change include, "
        "among others, intense droughts, water scarcity, severe fires, rising sea levels, "
        "flooding, melting polar ice, catastrophic storms and declining biodiversity."
    ),
    "Redes Neuronales": (
        "A neural network is a machine learning program, or model, that makes decisions in "
        "a manner similar to the human brain, by using processes that mimic the way biological "
        "neurons work together to identify phenomena, weigh options and arrive at conclusions. "
        "Every neural network consists of layers of nodes, or artificial neurons — an input "
        "layer, one or more hidden layers, and an output layer. Each node connects to others, "
        "and has its own associated weight and threshold. If the output of any individual node "
        "is above the specified threshold value, that node is activated, sending data to the "
        "next layer of the network. Otherwise, no data is passed along to the next layer of "
        "the network. Neural networks rely on training data to learn and improve their accuracy "
        "over time. Once the learning algorithms are fine-tuned, they are powerful tools in "
        "computer science and artificial intelligence, allowing us to classify and cluster data "
        "at a high velocity. Tasks in speech recognition or image recognition can take minutes "
        "versus hours when compared to the manual identification by human experts."
    ),
}


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class T5Model:
    # Wrapper sobre T5ForConditionalGeneration de HuggingFace Transformers.
    #
    # Gestiona tres responsabilidades bien delimitadas:
    #   1. Carga del tokenizador SentencePiece y de los pesos preentrenados
    #      del modelo encoder-decoder T5.
    #   2. Generación de resúmenes a partir de texto en inglés (método summarize).
    #      Internamente aplica beam search sobre el espacio de tokens.
    #   3. Extracción de la matriz de atención cruzada entre encoder y decoder
    #      para su visualización como heatmap (método get_cross_attention).

    def __init__(self, model_id: str = "t5-small"):
        # --- Selección del dispositivo de cómputo ---
        # Se prueban los aceleradores en orden de preferencia:
        #   CUDA : GPU NVIDIA. Disponible en Windows / Linux con driver CUDA.
        #          Ofrece la mayor aceleración; necesario para modelos grandes.
        #   MPS  : Metal Performance Shaders, GPU integrada de Apple Silicon
        #          (chips M1, M2, M3). Disponible en macOS >= 12.3 con PyTorch >= 1.13.
        #   CPU  : fallback universal. Funciona en cualquier equipo pero es
        #          significativamente más lento para modelos de +200M parámetros.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model_id = model_id

        # --- Tokenizador ---
        # AutoTokenizer identifica automáticamente el tipo de tokenizador
        # asociado al modelo (T5 usa SentencePiece, un algoritmo de tokenización
        # a nivel de subpalabras que divide palabras en unidades menores
        # cuando no están en el vocabulario base de ~32,000 tokens).
        # Primera ejecución: descarga ~800 KB de archivos de vocabulario.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # --- Pesos del modelo ---
        # T5ForConditionalGeneration instancia la arquitectura encoder-decoder
        # completa (embeddings, capas de atención, FFN, head de salida) y carga
        # los pesos preentrenados desde HuggingFace Hub.
        # Primera ejecución: descarga entre 60 MB (t5-small) y 900 MB (t5-base).
        self.model = T5ForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device)

        # --- Modo evaluación ---
        # model.eval() desactiva el dropout y las capas de BatchNorm que solo
        # se usan durante el entrenamiento. Combinado con torch.no_grad() en
        # cada forward pass, reduce el uso de memoria y acelera la inferencia.
        self.model.eval()

    # -----------------------------------------------------------------------
    # Método 1: generación de resumen (inferencia principal)
    # -----------------------------------------------------------------------

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
    ) -> dict:
        # Genera un resumen en inglés a partir del texto de entrada.
        #
        # Parámetros:
        #   text                 : texto a resumir (solo inglés recomendado)
        #   max_length           : máximo de tokens en la salida del decoder
        #   min_length           : mínimo de tokens en la salida del decoder
        #   num_beams            : hipótesis que beam search mantiene en paralelo
        #                          (1 = greedy decoding, >1 mejora calidad)
        #   length_penalty       : exponente aplicado a la longitud de la secuencia
        #                          al puntuar las hipótesis del beam search.
        #                          >1.0 favorece resúmenes más largos;
        #                          <1.0 favorece resúmenes más cortos.
        #   no_repeat_ngram_size : evita que el modelo repita n-gramas del mismo
        #                          tamaño en la salida (reduce frases duplicadas)
        #
        # Retorna dict con: summary, input_tokens, output_tokens,
        #                   compression_ratio, elapsed_seconds, device.

        # Paso 1: prefijo de tarea.
        # El framework text-to-text de T5 distingue tareas mediante un prefijo
        # de texto en la entrada. "summarize: " instruye al modelo a producir
        # un resumen. El mismo modelo usaría "translate English to French: " para
        # traducción o "cola sentence: " para clasificación gramatical, sin
        # ningún cambio en los pesos.
        prefixed_text = "summarize: " + text.strip()

        # Paso 2: tokenización del texto de entrada.
        # El tokenizador SentencePiece convierte el texto en una secuencia de
        # IDs enteros que representan subpalabras del vocabulario del modelo.
        # truncation=True recorta la secuencia si supera 512 tokens, que es el
        # límite del encoder en la configuración estándar de T5.
        # return_tensors="pt" devuelve tensores de PyTorch listos para el modelo.
        inputs = self.tokenizer(
            prefixed_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        # Paso 3: generación autoregresiva con beam search.
        # El decoder produce un token a la vez. En cada paso:
        #   a) Aplica masked self-attention sobre los tokens ya generados.
        #   b) Aplica cross-attention sobre la salida del encoder para "leer"
        #      el texto de entrada (aquí es donde Q viene del decoder y K,V
        #      vienen del encoder).
        #   c) Aplica una capa FFN y proyecta al espacio del vocabulario.
        #   d) Selecciona el token siguiente según las puntuaciones de beam search.
        # torch.no_grad() desactiva el cálculo del grafo de diferenciación,
        # reduciendo el consumo de memoria RAM/VRAM durante la inferencia.
        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,             # detiene cuando todos los beams llegan a </s>
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        elapsed = round(time.time() - t0, 3)

        # Paso 4: decodificación de la salida.
        # Convierte la secuencia de IDs generados por el decoder de vuelta
        # a texto legible. skip_special_tokens=True elimina tokens de control
        # como <pad>, </s> y <unk> del resultado final.
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        n_input  = int(inputs.input_ids.shape[1])
        n_output = int(output_ids.shape[1])

        return {
            "summary":           summary,
            "input_tokens":      n_input,
            "output_tokens":     n_output,
            # ratio = tokens_entrada / tokens_salida. Valor > 1 indica compresión.
            "compression_ratio": round(n_input / max(n_output, 1), 2),
            "elapsed_seconds":   elapsed,
            "device":            str(self.device),
        }

    # -----------------------------------------------------------------------
    # Método 2: extracción de pesos de atención cruzada
    # -----------------------------------------------------------------------

    def get_cross_attention(
        self,
        input_text: str,
        output_text: str,
        max_enc_tokens: int = 80,
        max_dec_tokens: int = 40,
    ) -> dict:
        # Extrae y devuelve la matriz de atención cruzada de la última capa
        # del decoder para un par (entrada, salida) dado.
        #
        # La atención cruzada (cross-attention) es el mecanismo central que
        # permite al decoder consultar el encoder en cada paso de generación.
        # Sus tres tensores son:
        #   Q (Query)  : generado por el decoder. Representa la pregunta
        #                "¿qué información necesito para producir este token?"
        #   K (Key)    : generado por el encoder. Representa el índice de
        #                búsqueda "¿qué información hay disponible en la entrada?"
        #   V (Value)  : generado por el encoder. Contiene el contenido real
        #                que se extrae una vez que K responde a Q.
        #
        # El cálculo es: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        # donde d_k es la dimensión de las cabezas de atención.
        #
        # Parámetros:
        #   input_text      : texto de entrada original (sin el prefijo)
        #   output_text     : resumen ya generado (del método summarize)
        #   max_enc_tokens  : límite de tokens del encoder (controla ancho del heatmap)
        #   max_dec_tokens  : límite de tokens del decoder (controla alto del heatmap)
        #
        # Retorna dict con: attention (numpy 2D), encoder_tokens, decoder_tokens.

        prefixed = "summarize: " + input_text.strip()

        # Tokenizar la entrada y la salida por separado con límites reducidos.
        # Los límites controlan el tamaño del heatmap: demasiados tokens hacen
        # la visualización ilegible. 60 tokens de encoder y 30 de decoder
        # suele ser un buen balance para textos de demostración.
        enc = self.tokenizer(
            prefixed, return_tensors="pt",
            max_length=max_enc_tokens, truncation=True,
        ).to(self.device)

        dec = self.tokenizer(
            output_text, return_tensors="pt",
            max_length=max_dec_tokens, truncation=True,
        ).to(self.device)

        # Forward pass completo del modelo con output_attentions=True.
        # En lugar de generar tokens nuevos (como en summarize), aquí se pasa
        # directamente el resumen ya generado como decoder_input_ids para
        # obtener los pesos de atención que el modelo asignaría al procesarlo.
        # output_attentions=True hace que el modelo devuelva los pesos de todas
        # las capas y cabezas del encoder y del decoder (incluyendo cross-attention).
        with torch.no_grad():
            outputs = self.model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                decoder_input_ids=dec.input_ids,
                output_attentions=True,
                return_dict=True,
            )

        # outputs.cross_attentions es una tupla de tensores, uno por capa del decoder.
        # Forma de cada tensor: (batch_size=1, num_heads, dec_seq_len, enc_seq_len)
        #
        # Se selecciona la última capa [-1] porque es la más cercana a la generación
        # del token de salida y refleja las decisiones de atención más refinadas.
        # Se toma el primer (y único) ejemplo del batch [0] para obtener
        # un tensor de forma (num_heads, dec_seq_len, enc_seq_len).
        # Se promedian las cabezas con mean(dim=0) para obtener una sola matriz
        # 2D (dec_seq_len, enc_seq_len) visualizable como heatmap.
        last_cross_attn = outputs.cross_attentions[-1][0]        # (heads, dec, enc)
        attn_matrix = last_cross_attn.mean(dim=0).cpu().numpy()  # (dec, enc)

        # Convertir IDs numéricos a texto legible para las etiquetas de los ejes.
        # SentencePiece usa el prefijo "▁" (espacio subrayado) para marcar el
        # inicio de una nueva palabra, lo que permite reconstruir espacios
        # al decodificar una secuencia de subpalabras.
        enc_tokens = [
            self.tokenizer.convert_ids_to_tokens([tid])[0]
            for tid in enc.input_ids[0].tolist()
        ]
        dec_tokens = [
            self.tokenizer.convert_ids_to_tokens([tid])[0]
            for tid in dec.input_ids[0].tolist()
        ]

        return {
            "attention":      attn_matrix,  # numpy array de forma (dec_len, enc_len)
            "encoder_tokens": enc_tokens,   # lista de strings para el eje X del heatmap
            "decoder_tokens": dec_tokens,   # lista de strings para el eje Y del heatmap
        }
