FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ssh_server.py .

# Run unprivileged. /data is created here with the right owner so an empty
# named volume mounted over it inherits these permissions on first use.
RUN useradd --system --uid 10001 --no-create-home honeypot \
    && mkdir -p /data \
    && chown honeypot:honeypot /data

USER honeypot

EXPOSE 2222

CMD ["python", "ssh_server.py"]
