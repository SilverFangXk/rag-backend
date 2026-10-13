# Optional: for any host that runs containers (Fly.io, Railway, Render in Docker mode...).
# Not needed for Vercel, nor for the default Render setup (render.yaml uses Render's Python runtime).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /srv

# dependencies first: this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

# never run as root inside the container
RUN useradd --create-home appuser
USER appuser

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
