# hash:sha256:1add3c96fdbfacbbc3ffeda76e5e03135a878f4a93e7961d1c2e6a8018f81f49
FROM registry.codeocean.com/codeocean/miniconda3:4.8.2-python3.8-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgtk2.0-dev=2.24.32-1ubuntu1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    keras==2.4.3 \
    matplotlib==3.4.1 \
    numpy==1.20.2 \
    opencv-python==3.4.8.29 \
    pandas==1.2.4 \
    pillow==8.2.0 \
    scikit-image==0.18.1 \
    scipy==1.6.2 \
    sklearn==0.0 \
    tensorflow==2.4.1

COPY postInstall /
RUN /postInstall
