# Base image with CUDA support
FROM nvidia/cuda:12.1.105-cudnn8-runtime-ubuntu22.04

# Set workdir
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy app
COPY ./app ./app
COPY ./model ./model

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
