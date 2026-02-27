🔌 Documentación de Endpoints
La API cuenta con 6 endpoints funcionales. Los principales son:

🧠 Modelos Propios (Scikit-Learn)
POST /predict: Recibe un texto (reseña), lo limpia y utiliza el modelo Random Forest entrenado para predecir si el sentimiento es Positivo o Negativo, devolviendo también el nivel de confianza de la predicción.

GET /model-info: Devuelve los metadatos y la configuración interna del modelo de Scikit-Learn (hiperparámetros, tamaño del vocabulario, etc.).

🤖 Modelos Pre-entrenados (HuggingFace)
POST /hf-sentiment: Un endpoint alternativo que utiliza el modelo de estado del arte distilbert-base-uncased para realizar un análisis de sentimiento avanzado.

POST /hf-generate: Generador de texto integrado con gpt2. Recibe un prompt (texto inicial) y devuelve una continuación lógica generada por la IA.

☁️ Evidencia del Despliegue en la Nube
Para cumplir con la parte opcional de la práctica, la API fue correctamente empaquetada en Docker y desplegada en Google Cloud Run.

Nota importante sobre FinOps: Para evitar costes innecesarios de facturación en la tarjeta de crédito asociada a Google Cloud, la infraestructura en vivo ha sido eliminada tras comprobar su correcto funcionamiento.

📸 Todas las pruebas del despliegue exitoso (interfaz Swagger pública y peticiones HTTP completadas) se encuentran en la carpeta /capturas_cloud/ de este repositorio.

🛠️ Stack Tecnológico
Machine Learning: Scikit-Learn, Pandas, Joblib.

Deep Learning & NLP: HuggingFace Transformers, PyTorch.

MLOps: MLflow.

Backend: FastAPI, Uvicorn, Pydantic.

DevOps & Nube: Docker, Google Cloud Platform (Artifact Registry, Cloud Run).
