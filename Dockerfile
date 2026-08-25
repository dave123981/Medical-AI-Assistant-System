FROM python:3.11-slim

WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

ENV MODEL_DIR=/service/models
ENV MODEL_FILENAME=v1_decision_tree.joblib
ENV MODEL_VERSION=v1-decision-tree

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
