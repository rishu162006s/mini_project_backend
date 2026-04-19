# Base Python image (stable for TensorFlow)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for numpy/tensorflow/scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Render requires listening on this port
EXPOSE 10000

# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
