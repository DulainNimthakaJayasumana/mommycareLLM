
FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x /start.sh

EXPOSE 8080

CMD ["/bin/sh", "/start.sh"]
