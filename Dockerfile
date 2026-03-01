# Usamos una imagen oficial de Python ligera
FROM python:3.11-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos primero el archivo de dependencias 
COPY requirements.txt .

# Instalamos las dependencias sin guardar caché para que la imagen pese menos
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de los archivos (main.py, model_rf.joblib)
COPY . .

# Exponemos el puerto que usará FastAPI
EXPOSE 8080

# Comando para ejecutar la API cuando arranque el contenedor

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
