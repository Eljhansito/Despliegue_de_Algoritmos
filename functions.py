import re
import string
import joblib


def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Limpia el texto eliminando puntuación y espacios extra.

    Args:
        text (str): Texto de entrada.
        lowercase (bool): Si True, convierte a minúsculas.

    Returns:
        str: Texto limpio.
    """
    if lowercase:
        text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_sentiment(model, text: str, clean: bool = True) -> dict:
    """
    Realiza la predicción de sentimiento usando el pipeline de sklearn.

    Args:
        model: Pipeline de sklearn cargado (vectorizador + clasificador).
        text (str): Texto de la reseña a clasificar.
        clean (bool): Si True, limpia el texto antes de predecir.

    Returns:
        dict: Diccionario con 'sentiment', 'prediction_code' y 'confidence'.
    """
    if clean:
        text = clean_text(text)
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    return {
        "sentiment": "Positivo" if pred == 1 else "Negativo",
        "prediction_code": int(pred),
        "confidence": round(float(max(proba)), 4)
    }


def get_model_info(model) -> dict:
    """
    Extrae metadatos del pipeline sklearn (vectorizador + clasificador).

    Args:
        model: Pipeline de sklearn con pasos 'vectorizer' y 'classifier'.

    Returns:
        dict: Información del modelo: tipo, hiperparámetros y vocabulario.
    """
    classifier = model.named_steps['classifier']
    vectorizer = model.named_steps['vectorizer']
    vocab_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else None
    return {
        "model_type": type(classifier).__name__,
        "n_estimators": classifier.n_estimators,
        "max_features_tfidf": vectorizer.max_features,
        "vocabulary_size": vocab_size,
        "stop_words": "english"
    }


def load_model(model_path: str):
    """
    Carga un modelo guardado con joblib.

    Args:
        model_path (str): Ruta al archivo .joblib del modelo.

    Returns:
        Pipeline sklearn cargado.
    """
    return joblib.load(model_path)
