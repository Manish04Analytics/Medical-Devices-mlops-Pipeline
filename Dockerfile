# ============================================================
# MEDICAL DEVICE DEMAND PREDICTION API
# Docker Production Image
# ============================================================

# Use a lightweight Python image
FROM python:3.11-slim

# Prevent Python from creating .pyc files
# and ensure logs appear immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# ------------------------------------------------------------
# Install system dependencies
# ------------------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Copy dependency file
# ------------------------------------------------------------

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Copy application
# ------------------------------------------------------------

COPY src ./src

# Copy production model
COPY outputs ./outputs

# ------------------------------------------------------------
# Expose FastAPI port
# ------------------------------------------------------------

EXPOSE 8000

# ------------------------------------------------------------
# Start FastAPI
# ------------------------------------------------------------

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]