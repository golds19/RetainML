FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit>=1.53.1 \
    requests>=2.32.5 \
    pandas>=2.3.3

COPY . ./src/dashboard/

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true"]
