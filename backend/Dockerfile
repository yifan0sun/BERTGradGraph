FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && apt-get clean

# Set working directory to root of copied repo
WORKDIR /

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from the repo root into the container root
COPY . .

# Expose port for Hugging Face
EXPOSE 7860

# ✅ Launch server.py's FastAPI app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]