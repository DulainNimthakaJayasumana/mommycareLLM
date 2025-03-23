FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Create start script to run both applications
RUN echo '#!/bin/bash\npython api.py & python telegram_bot.py\nwait' > /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8080

CMD ["/bin/bash", "/app/start.sh"]