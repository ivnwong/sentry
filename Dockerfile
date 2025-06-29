FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY model_files/ ./model_files/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV ADMIN_TOKEN=your_admin_token_here

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
