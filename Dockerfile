# ============================
# 1. Use official Miniconda base image
# ============================
FROM continuumio/miniconda3

# ============================
# 2. Set working directory
# ============================
WORKDIR /app

# ============================
# 3. Create a Conda environment with RDKit
# ============================
RUN conda install -y -c conda-forge rdkit

# ============================
# 4. Copy requirements separately (cache-friendly)
# ============================
COPY requirements.txt .

# Install pip dependencies inside the same environment
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ============================
# 5. Copy the entire project
# ============================
COPY . .

# ============================
# 6. Expose API port
# ============================
EXPOSE 10000

# ============================
# 7. Run FastAPI with Uvicorn
# ============================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
