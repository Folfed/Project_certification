# =============================================================================
# 🐳 DOCKERFILE - API de Prédiction du Rendement du Maïs
# =============================================================================
# Multi-stage build pour une image optimisée

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Équipe Data Science"
LABEL description="API de prédiction du rendement du maïs en Afrique"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Copie des wheels depuis le builder
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copie du code source
COPY . .

# Création d'un utilisateur non-root pour la sécurité
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Exposition du port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Commande de démarrage
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
