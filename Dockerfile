# ============================
# 1. Base image with full RDKit
# ============================
FROM rdkit/rdkit:latest

# ============================
# 2. Set working directory
# ============================
WORKDIR /app

# ============================
# 3. Copy requirements file
#    (better build caching)
# ============================
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# ============================
# 4. Copy project files
# ============================
COPY . .

# ============================
# 5. Expose port for Render
# ============================
EXPOSE 10000

# ============================
# 6. Start FastAPI app
# ============================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
