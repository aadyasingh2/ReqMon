FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration and source files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY data/ ./data/

# Install Python dependencies and package
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "ramma_nlp.interpreter:app", "--host", "0.0.0.0", "--port", "8000"]
