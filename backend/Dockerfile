FROM python:3.10

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD bash -c "uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}"
