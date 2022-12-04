FROM python:3.10

WORKDIR /server

RUN apt-get update
COPY requirements.txt /server/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN apt install -y default-jre

COPY . .

RUN chmod +x /server/entrypoint_server.sh
EXPOSE 80

ENTRYPOINT python3 demo.py