🔌 Endpoints de la API

La API expone varios endpoints documentados interactivamente vía Swagger UI (/docs). Los más destacados incluyen:

[POST] /predict: Análisis de sentimiento utilizando el modelo propio entrenado (Random Forest + TF-IDF).

[GET] /model-info: Extracción de metadatos e hiperparámetros del modelo base.

[POST] /hf-sentiment: Inferencia de sentimiento utilizando un pipeline pre-entrenado de HuggingFace (distilbert).

[POST] /hf-generate: Generación de texto (continuación de prompts) utilizando HuggingFace (gpt2).

☁️ Despliegue en Vivo
La API ha sido desplegada con éxito en Google Cloud Run 

Las imagenes estan en la carpeta "Capturas_Cloud"

🛠️ Tecnologías Utilizadas
Machine Learning: Scikit-Learn, Pandas, Joblib.

Deep Learning / NLP: HuggingFace Transformers, PyTorch.

MLOps & Tracking: MLflow.

Desarrollo Backend: FastAPI, Uvicorn, Pydantic.

DevOps & Cloud: Docker, Google Cloud Shell, GCP Artifact Registry, GCP Cloud Run.

