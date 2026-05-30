# inference.py
# ------------
# Responsabilidad única: toda la lógica relacionada con el modelo T5.
# Carga de pesos desde HuggingFace Hub, tokenización con SentencePiece,
# generación de resúmenes con beam search, cálculo opcional de ROUGE,
# benchmarking entre variantes y extracción configurable de pesos de
# atención cruzada para visualización en la interfaz.
#
# Este módulo no importa Streamlit ni conoce la UI. Solo recibe datos,
# ejecuta el modelo y devuelve resultados en estructuras Python puras (dict).

import time
from typing import Optional, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

try:
    from rouge_score import rouge_scorer
except ImportError:  # Permite que la app funcione aunque rouge-score no este instalado.
    rouge_scorer = None


# ---------------------------------------------------------------------------
# Catálogo de variantes del modelo
# ---------------------------------------------------------------------------
# t5-small / t5-base: modelos estándar preentrenados en C4 con span corruption.
#   Producen resúmenes coherentes. Recomendados para demos educativas en CPU.
#   t5-small  ~ 60M parámetros, t5-base ~ 220M parámetros.
#
# t5-efficient-*: variantes compactas diseñadas por Google para reducir costo.
#   Útiles cuando el tiempo de descarga o la memoria RAM son limitados.
#
# Los IDs de la derecha son los identificadores exactos en HuggingFace Hub.
# Al instanciar T5Model, los pesos se descargan una sola vez y quedan en
# caché local (~/.cache/huggingface/) para usos posteriores sin conexión.
AVAILABLE_MODELS = {
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    #"t5-efficient-tiny": "google/t5-efficient-tiny",
    #"t5-efficient-small": "google/t5-efficient-small",
    #"t5-efficient-base": "google/t5-efficient-base",
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
    "Cambio Climático": (
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


# Parámetros aproximados, útiles para mostrar información académica y comparativas.
MODEL_METADATA = {
    "t5-small": {"parameters": "~60M", "recommended_use": "Demo equilibrada en CPU"},
    "t5-base": {"parameters": "~220M", "recommended_use": "Mayor calidad, más lento"},
    "t5-efficient-tiny": {"parameters": "~16M", "recommended_use": "Muy rápido, menor calidad"},
    "t5-efficient-small": {"parameters": "~60M", "recommended_use": "Balance eficiencia/calidad"},
    "t5-efficient-base": {"parameters": "~220M", "recommended_use": "Eficiente grande"},
}


class T5Model:
    # Wrapper sobre T5ForConditionalGeneration de HuggingFace Transformers.

    def __init__(self, model_id: str = "t5-small"):
        # --- Selección del dispositivo de cómputo ---
        # CUDA : GPU NVIDIA.
        # MPS  : GPU de Apple Silicon.
        # CPU  : fallback universal.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    # -----------------------------------------------------------------------
    # Información del modelo cargado
    # -----------------------------------------------------------------------

    def get_model_info(self) -> dict:
        # Devuelve dimensiones reales del modelo cargado desde su config.
        cfg = self.model.config
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "d_model": getattr(cfg, "d_model", None),
            "d_ff": getattr(cfg, "d_ff", None),
            "d_kv": getattr(cfg, "d_kv", None),
            "num_heads": getattr(cfg, "num_heads", None),
            "num_layers": getattr(cfg, "num_layers", None),
            "num_decoder_layers": getattr(cfg, "num_decoder_layers", getattr(cfg, "num_layers", None)),
            "vocab_size": getattr(cfg, "vocab_size", None),
            "parameter_count_trainable": int(sum(p.numel() for p in self.model.parameters() if p.requires_grad)),
            "parameter_count_total": int(sum(p.numel() for p in self.model.parameters())),
        }

    # -----------------------------------------------------------------------
    # Método 1: generación de resumen
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
        prefixed_text = "summarize: " + text.strip()

        inputs = self.tokenizer(
            prefixed_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        elapsed = round(time.time() - t0, 3)

        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        n_input = int(inputs.input_ids.shape[1])
        n_output = int(output_ids.shape[1])

        result = {
            "summary": summary,
            "input_tokens": n_input,
            "output_tokens": n_output,
            "compression_ratio": round(n_input / max(n_output, 1), 2),
            "elapsed_seconds": elapsed,
            "device": str(self.device),
            "model_id": self.model_id,
        }

        if self.device.type == "cuda":
            result["cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
            result["cuda_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 2)

        return result

    # -----------------------------------------------------------------------
    # Método 2: ROUGE
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_rouge(reference_text: str, generated_text: str) -> dict:
        # Calcula ROUGE-1, ROUGE-2 y ROUGE-L contra un resumen de referencia.
        if rouge_scorer is None:
            raise ImportError(
                "No está instalado rouge-score. Ejecuta: pip install rouge-score"
            )

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )
        scores = scorer.score(reference_text, generated_text)

        return {
            metric: {
                "precision": round(value.precision, 4),
                "recall": round(value.recall, 4),
                "fmeasure": round(value.fmeasure, 4),
            }
            for metric, value in scores.items()
        }

    # -----------------------------------------------------------------------
    # Método 3: extracción configurable de pesos de atención cruzada
    # -----------------------------------------------------------------------

    def get_cross_attention(
        self,
        input_text: str,
        output_text: str,
        max_enc_tokens: int = 80,
        max_dec_tokens: int = 40,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
        average_layers: bool = False,
    ) -> dict:
        # Extrae una matriz de cross-attention configurable.
        # Forma interna de cada capa: (batch_size, num_heads, dec_seq_len, enc_seq_len)
        # layer_idx=None y average_layers=False -> última capa.
        # layer_idx=N -> capa N del decoder. average_layers=True -> promedio de capas.
        # head_idx=None -> promedio de cabezas. head_idx=N -> cabeza específica.
        prefixed = "summarize: " + input_text.strip()

        enc = self.tokenizer(
            prefixed,
            return_tensors="pt",
            max_length=max_enc_tokens,
            truncation=True,
        ).to(self.device)

        dec = self.tokenizer(
            output_text,
            return_tensors="pt",
            max_length=max_dec_tokens,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                decoder_input_ids=dec.input_ids,
                output_attentions=True,
                return_dict=True,
            )

        cross_attentions = outputs.cross_attentions
        num_layers = len(cross_attentions)
        sample = cross_attentions[0]
        num_heads = int(sample.shape[1])

        if average_layers:
            # Lista de tensores: cada uno (batch, heads, dec, enc).
            # stack -> (layers, batch, heads, dec, enc)
            selected = torch.stack(list(cross_attentions), dim=0).mean(dim=0)[0]
            selected_layer_label = "Promedio de capas"
        else:
            if layer_idx is None:
                layer_idx = num_layers - 1
            if not 0 <= layer_idx < num_layers:
                raise ValueError(f"layer_idx debe estar entre 0 y {num_layers - 1}")
            selected = cross_attentions[layer_idx][0]
            selected_layer_label = f"Layer {layer_idx}"

        if head_idx is None:
            attn_matrix = selected.mean(dim=0).detach().cpu().numpy()
            selected_head_label = "Promedio de heads"
        else:
            if not 0 <= head_idx < num_heads:
                raise ValueError(f"head_idx debe estar entre 0 y {num_heads - 1}")
            attn_matrix = selected[head_idx].detach().cpu().numpy()
            selected_head_label = f"Head {head_idx}"

        enc_tokens = [
            self.tokenizer.convert_ids_to_tokens([tid])[0]
            for tid in enc.input_ids[0].tolist()
        ]
        dec_tokens = [
            self.tokenizer.convert_ids_to_tokens([tid])[0]
            for tid in dec.input_ids[0].tolist()
        ]

        interpretation = self.interpret_attention(attn_matrix, enc_tokens)

        return {
            "attention": attn_matrix,
            "encoder_tokens": enc_tokens,
            "decoder_tokens": dec_tokens,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "selected_layer": selected_layer_label,
            "selected_head": selected_head_label,
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "average_layers": average_layers,
            "interpretation": interpretation,
        }

    # -----------------------------------------------------------------------
    # Método 4: interpretación automática del heatmap
    # -----------------------------------------------------------------------

    @staticmethod
    def _clean_token(token: str) -> str:
        # Limpia tokens SentencePiece para mostrarlos de forma legible.
        return (
            token.replace("▁", " ")
            .replace("</s>", "")
            .replace("<pad>", "")
            .strip()
        )

    @classmethod
    def interpret_attention(
        cls,
        attention_matrix: np.ndarray,
        encoder_tokens: Sequence[str],
        top_k: int = 8,
    ) -> dict:
        # Identifica los top_k tokens del encoder más consultados por el decoder.
        # Promedia la atención por columnas: mayor promedio = más consultado globalmente.
        if attention_matrix.size == 0:
            return {"top_tokens": [], "message": "No hay datos suficientes para interpretar."}

        col_importance = attention_matrix.mean(axis=0)
        candidates = []

        for idx, score in enumerate(col_importance):
            token = encoder_tokens[idx] if idx < len(encoder_tokens) else ""
            clean = cls._clean_token(token)
            if not clean or clean in {".", ",", ":", ";", "'", '"', "-"}:
                continue
            candidates.append(
                {
                    "index": int(idx),
                    "token": token,
                    "clean_token": clean,
                    "score": round(float(score), 6),
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        top_tokens = candidates[:top_k]
        readable = [item["clean_token"] for item in top_tokens[:5]]

        if readable:
            message = (
                "El decoder consultó con mayor intensidad los tokens de entrada: "
                + ", ".join(readable)
                + ". Estos tokens fueron los más influyentes dentro de la matriz de cross-attention seleccionada."
            )
        else:
            message = "No se detectaron tokens informativos destacados en la atención seleccionada."

        return {
            "top_tokens": top_tokens,
            "message": message,
        }
