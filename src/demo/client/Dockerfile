FROM python:3.9

WORKDIR /client

RUN apt-get update
COPY requirements.txt /client/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /client/entrypoint_client.sh
EXPOSE 8501

ENTRYPOINT [ "sh", "/client/entrypoint_client.sh" ]