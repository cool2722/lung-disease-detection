FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib2.0-0: runtime libs needed by opencv for image decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app.py .
# Optional: bring in a trained checkpoint if present at build time
# (drop it at weights/best.pt before building). Otherwise the app
# falls back to demo mode automatically.
COPY weights/ weights/

EXPOSE 7860
ENV PORT=7860

CMD ["python", "app.py"]
