FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary directories and files
COPY app/ app/
COPY src/ src/
COPY app/models/best_model.pkl app/models/
# Also copy preprocessing if separate, or ensure it's in models
# COPY app/models/preprocessing.pkl app/models/ 

# Expose API port
EXPOSE 8000

# Start API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
