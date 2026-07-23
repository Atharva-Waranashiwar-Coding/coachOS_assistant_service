FROM python:3.12-slim
WORKDIR /app
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt
COPY . .
RUN useradd --system --uid 10001 coachos && chown -R coachos:coachos /app
USER coachos
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
