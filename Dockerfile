# Runtime image for the voice screener. One uvicorn process serves the console,
# the browser and Twilio call sockets, and the post-call report.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# onnxruntime (Silero VAD) links libgomp; nothing else needs system packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml .
COPY server ./server
COPY static ./static
COPY plans ./plans
COPY models ./models

RUN useradd --system --uid 10001 --no-create-home app \
    && chown -R app:app /app
USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"

# --proxy-headers: behind a TLS-terminating proxy (Fly, Render, nginx) the app
# must see https/wss for the TwiML it hands to Twilio. Single process on
# purpose: metrics and reports are in-memory and per process.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
