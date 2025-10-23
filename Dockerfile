# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for psycopg2
RUN apt-get update && \
    apt-get install -y build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy app code and pickles
COPY quotations.py model.pkl label_encoders.pkl target_encodings.pkl selected_features.json ./


# Expose port
EXPOSE 80

# Run FastAPI app
CMD ["uvicorn","quotations:app", "--host", "0.0.0.0", "--port", "80"]