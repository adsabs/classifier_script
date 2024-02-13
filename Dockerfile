FROM python:3.8

WORKDIR /app

COPY requirements.txt /app
COPY scripts/entrypoint.sh /app/scripts/
# COPY dev-requirements.txt /app

RUN ls -lh 

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
    # pip install -r dev-requirements.txt

RUN chmod +x scripts/entrypoint.sh

CMD ["scripts/entrypoint.sh"]

