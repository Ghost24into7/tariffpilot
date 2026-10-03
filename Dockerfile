FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TP_ROOT=/app TP_STATE_DIR=/app/.state
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[api,gemini]"
COPY prompts ./prompts
COPY skills ./skills
COPY policy ./policy
COPY data ./data
RUN useradd -m app && mkdir -p /app/.state && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/healthz')" || exit 1
CMD ["uvicorn", "tariffpilot.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
