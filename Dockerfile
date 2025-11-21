# ============================
# 1. Use a maintained RDKit image
# ============================
FROM inocybe/rdkit:latest

# ============================
# 2. Set working directory
# ============================
WORKDIR /app

# ============================
# 3. Copy requirements file
# ============================
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================
# 4. Copy project files
# ============================
COPY . .

# ============================
# 5. Expose port
# ============================
EXPOSE 10000

# ============================
# 6. Start FastAPI
# ============================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
