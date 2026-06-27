FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces expects the service to listen on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "product.api_server:app", "--host", "0.0.0.0", "--port", "7860"]
