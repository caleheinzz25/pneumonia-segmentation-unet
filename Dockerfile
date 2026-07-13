# Dockerfile
# Gunakan image resmi Python slim
FROM python:3.10-slim

# Install system dependencies yang dibutuhkan oleh OpenCV, Pydicom, dan Git
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Tentukan direktori kerja di dalam kontainer
WORKDIR /app

# Salin file requirements.txt terlebih dahulu
COPY requirements.txt .

# Install dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua source code ke dalam kontainer
COPY . .

# Ekspos port Gradio (sesuai config.yaml)
EXPOSE 7860

# Jalankan aplikasi wrapper app.py
CMD ["python", "app.py"]
