# ============================
# 1. RDKit base image (GHCR, reliable on Render)
# ============================
FROM ghcr.io/mcs07/rdkit:latest

# ============================
# 2. Set app directory
# ============================
WORKDIR /app

# ============================
# 3. Install dependencies
# ============================
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ============================
# 4. Copy the entire backend
# ============================
COPY . .

# ============================
# 5. Expose port for FastAPI
# ============================
EXPOSE 10000

# ============================
# 6. Run FastAPI server
# ============================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
