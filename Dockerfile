FROM python:3.12-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements-prod.txt .
RUN pip wheel --wheel-dir=/wheels -r requirements-prod.txt

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN groupadd --system coachos && useradd --system --gid coachos --home /app coachos
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-prod.txt && rm -rf /wheels
COPY --chown=coachos:coachos . .
USER coachos
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
