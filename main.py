from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
from transformers import pipeline as hf_pipeline

from functions import predict_sentiment, get_model_info, clean_text

# ============================================================
# 1. Inicializar la aplicación FastAPI
# ============================================================
app = FastAPI(
    title="API de Análisis de Sentimiento Musical",
    description=(
        "API que combina un modelo Random Forest propio con pipelines de HuggingFace "
        "para análisis de sentimiento y generación de texto."
    ),
    version="2.0.0"
)

# ============================================================
# 2. Cargar el modelo sklearn al arrancar
# ============================================================
try:
    model = joblib.load('model_rf.joblib')
    print("Modelo sklearn cargado correctamente.")
except Exception as e:
    print(f"Advertencia: no se pudo cargar el modelo sklearn: {e}")
    model = None

# ============================================================
# 3. Lazy loading de los pipelines de HuggingFace
#    (se cargan la primera vez que se llama al endpoint)
# ============================================================
_hf_sentiment_pipeline = None
_hf_generator_pipeline = None


def get_hf_sentiment():
    """Carga (una sola vez) el pipeline de sentimiento de HuggingFace."""
    global _hf_sentiment_pipeline
    if _hf_sentiment_pipeline is None:
        print("Cargando pipeline HF de sentimiento...")
        _hf_sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _hf_sentiment_pipeline


def get_hf_generator():
    """Carga (una sola vez) el pipeline de generación de texto de HuggingFace."""
    global _hf_generator_pipeline
    if _hf_generator_pipeline is None:
        print("Cargando pipeline HF de generación de texto...")
        _hf_generator_pipeline = hf_pipeline(
            "text-generation",
            model="gpt2"
        )
    return _hf_generator_pipeline


# ============================================================
# 4. Esquemas de entrada (Pydantic)
# ============================================================
class ReviewRequest(BaseModel):
    reviewText: str


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 60


# ============================================================
# ENDPOINTS
# ============================================================

# --- Endpoint 1: Health check ---
@app.get("/", tags=["General"])
def health_check():
    """Comprueba que la API está en funcionamiento."""
    return {
        "status": "ok",
        "message": "API de Análisis de Sentimiento Musical funcionando correctamente.",
        "version": "2.0.0"
    }


# --- Endpoint 2: Información del modelo sklearn ---
@app.get("/model-info", tags=["Sklearn"])
def model_info():
    """Devuelve los metadatos del pipeline sklearn (Random Forest + TF-IDF)."""
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo sklearn no está disponible.")
    return get_model_info(model)


# --- Endpoint 3: Predicción rápida via URL (GET) ---
@app.get("/predict/{text}", tags=["Sklearn"])
def predict_quick(text: str):
    """
    Predice el sentimiento de una reseña pasada directamente en la URL.
    Útil para pruebas rápidas desde el navegador.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo sklearn no está disponible.")
    cleaned = clean_text(text)
    return predict_sentiment(model, cleaned, clean=False)


# --- Endpoint 4: Predicción completa (POST) ---
@app.post("/predict", tags=["Sklearn"])
def predict(request: ReviewRequest):
    """
    Predice el sentimiento de una reseña musical usando el modelo Random Forest propio.
    Devuelve la etiqueta (Positivo/Negativo), el código numérico y la confianza.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo sklearn no está disponible.")
    return predict_sentiment(model, request.reviewText)


# --- Endpoint 5: Sentimiento con HuggingFace (Pipeline HF #1) ---
@app.post("/hf-sentiment", tags=["HuggingFace"])
def hf_sentiment_endpoint(request: ReviewRequest):
    """
    Analiza el sentimiento usando el pipeline de HuggingFace
    (DistilBERT fine-tuned en SST-2).
    Devuelve la etiqueta POSITIVE/NEGATIVE y la puntuación de confianza.
    """
    try:
        classifier = get_hf_sentiment()
        # Truncar a 512 tokens para respetar el límite del modelo
        result = classifier(request.reviewText[:512])[0]
        return {
            "text": request.reviewText[:100] + ("..." if len(request.reviewText) > 100 else ""),
            "label": result["label"],
            "score": round(result["score"], 4),
            "model": "distilbert-base-uncased-finetuned-sst-2-english"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en pipeline HF: {str(e)}")


# --- Endpoint 6: Generación de texto con HuggingFace (Pipeline HF #2) ---
@app.post("/hf-generate", tags=["HuggingFace"])
def hf_generate_endpoint(request: GenerateRequest):
    """
    Genera una continuación de texto a partir de un prompt usando GPT-2.
    Parámetro opcional: max_new_tokens (tokens nuevos a generar, default 60).
    """
    try:
        generator = get_hf_generator()
        result = generator(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=50256  # EOS token de GPT-2
        )[0]
        generated = result["generated_text"]
        return {
            "prompt": request.prompt,
            "generated_text": generated,
            "new_tokens": len(generated.split()) - len(request.prompt.split()),
            "model": "gpt2"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en pipeline HF: {str(e)}")
