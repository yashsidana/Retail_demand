FROM python:3.10-slim

WORKDIR /app

# Install system build dependencies for C++ compilers (CBC solver, XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    coinor-cbc \
    coinor-libcbc-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

