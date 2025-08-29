# -------- Base image (slim for small size) --------
FROM python:3.11-slim

# -------- Workdir --------
WORKDIR /app

# -------- System deps (only what's needed) --------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# -------- Copy and install Python deps first (leverage Docker layer cache) --------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# -------- Copy the rest of your application code --------
COPY . .

# -------- Security: run as non-root user --------#
RUN useradd -m appuser
USER appuser

# -------- Expose the app port --------
EXPOSE 8080

# -------- Start command (production) --------
CMD ["gunicorn", "-w", "2", "-k", "gthread", "--timeout", "30", "-b", "0.0.0.0:8080", "serverGatewayFlaskAi:app"]
