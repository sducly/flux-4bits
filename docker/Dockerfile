FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa -y && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    espeak \
    espeak-ng \
    festival \
    ffmpeg \
    g++ \
    git \
    gpg-agent \
    libgtk-3-dev \
    libsm6 \
    libsndfile1 \
    libstdc++6 \
    libvulkan1 \
    mbrola \
    mbrola-fr2 \
    pkg-config \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    software-properties-common \
    unzip \
    webkit2gtk-4.0-dev \
    wget \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

#RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
#    python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -U xformers --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 

CMD ["tail", "-f", "/dev/null"]