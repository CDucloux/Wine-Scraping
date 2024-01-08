FROM python:3.10-slim-buster
WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install poetry \ 
    && poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY streamlit_app.py .
COPY src ./src
COPY data ./data
COPY img ./img

RUN addgroup --system app && adduser --system --group app
USER app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]