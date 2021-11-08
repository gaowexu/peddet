FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install dependencies
RUN apt-get -y update && \
    apt-get install -y nginx \
    wget \
    git \
    unzip \
    cmake \
    python3 \
    python3-opencv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip==21.3.1
RUN pip3 install --no-cache Cython==0.29.24 \
    tensorflow-gpu==2.5.0 \
    onnx==1.4.1 \
    numpy==1.19.5 \
    flask==1.1.2 \
    gevent==20.12.1 \
    gunicorn==20.0.4

# Download pedestrian properties recognition model
RUN mkdir -p /opt/ml/model
RUN wget -c https://workshop-anker.s3.amazonaws.com/models/multi_tasks_classifier_models.zip -O /opt/ml/model/multi_tasks_classifier_models.zip
RUN cd /opt/ml/model/ && unzip multi_tasks_classifier_models.zip && rm -rf multi_tasks_classifier_models.zip

# Copy source code to container
COPY detector /opt/program
RUN chmod +x /opt/program/serve
WORKDIR /opt/program

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

ENTRYPOINT ["python3", "serve"]

