# Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for matplotlib and fonts
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs and charts directories
RUN mkdir -p /app/logs /app/charts

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLCONFIGDIR=/tmp/matplotlib

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('https://open-api.bingx.com/openApi/swap-v2/quote/ticker', timeout=5)" || exit 1

# Run bot
CMD ["python", "main.py"]
