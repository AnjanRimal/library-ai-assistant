FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY library_assistant ./library_assistant
RUN pip install --no-cache-dir .

RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["uvicorn", "library_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
