FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    g++ \
    make \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopencv-dev \
    && apt-get purge -y 'python3-numpy*' 'python3-matplotlib*' 'python3-pil*' || true \
    && apt-get autoremove -y \
    && rm -rf /usr/lib/python3/dist-packages/numpy* \
              /usr/lib/python3/dist-packages/matplotlib* \
              /usr/lib/python3/dist-packages/mpl_toolkits* \
              /usr/lib/python3/dist-packages/PIL* \
              /usr/lib/python3/dist-packages/Pillow* \
              /var/lib/apt/lists/*




WORKDIR /app

COPY src/requirements.txt /app/src/requirements.txt
RUN pip3 install --no-cache \
    -r /app/src/requirements.txt \
    numpy matplotlib Pillow

#pip install -r requirements.txt --break-system-packages
COPY data/ /app/data/
COPY src/ /app/src/

RUN cd /app/src && make

WORKDIR /app/src

# Uso:
#   Paso 1 (Python -> binario):  docker run <img> python3 utils.py 1
#   Paso 2 (C++ Strassen):       docker run <img> ./strassen_app
#   Paso 3 (visualización):      docker run <img> python3 utils.py 2
#Que se ejecute quede ejecutandose

#docker build -t pam-teoria .
#docker run -it --rm --gpus all pam-teoria bash
CMD bash -c " tail -f /dev/null"  
