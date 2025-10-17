FROM python:3.11-slim

WORKDIR /app

# OpenMP runtime needed by scikit-learn/SHAP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050

CMD ["gunicorn", "main:server", "-b", "0.0.0.0:8050"]