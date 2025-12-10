FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# 設置環境變數
ENV PYTHON_VERSION=3.8
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

# 安裝基本依賴和 OpenCV 依賴庫以及解壓工具
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    python${PYTHON_VERSION}-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unzip \
    wget \
    locales \
    && rm -rf /var/lib/apt/lists/*

# 設置語言環境
RUN locale-gen en_US.UTF-8
RUN locale-gen zh_TW.UTF-8
ENV LANG=zh_TW.UTF-8
ENV LANGUAGE=zh_TW:zh
ENV LC_ALL=zh_TW.UTF-8

# 設置 Python 3.8 為預設版本
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 安裝 PyTorch 和 CUDA 支援
RUN pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

# 複製模型和資料（使用壓縮文件）
# COPY draw_trainingdata.zip .

# 解壓縮模型檔案，然後刪除 zip 檔以節省空間
# RUN unzip -O UTF-8 draw_trainingdata.zip && rm draw_trainingdata.zip

# 安裝 gdown 並下載、解壓縮資料
RUN pip install gdown && \
    gdown 13YjzEWZwkNvKrpD6Mp8xaAgLFJ0IwCaZ && \
    unzip draw_trainingdata.zip && rm draw_trainingdata.zip

# 複製 API 代碼
COPY api.py .
COPY autoScore.py .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 