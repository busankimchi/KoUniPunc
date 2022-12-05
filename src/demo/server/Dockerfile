FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

WORKDIR /server

# install python3.9
RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get install -y curl
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install -y --no-install-recommends \
    python3.9 python3.9-venv python3.9-dev

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

RUN apt-get install -y --no-install-recommends \
    python3-setuptools python3.9-distutils

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install Dependencies of Miniconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# copy all necessary files
COPY . .

RUN /bin/bash -c "conda env create --file env.yaml"
RUN echo "conda activate punc-venv" >> ~/.bashrc
RUN /bin/bash -c "conda activate punc-venv"

RUN chmod +x /server/src/demo/server/entrypoint_server.sh
EXPOSE 8080

ENTRYPOINT [ "sh", "/server/entrypoint_server.sh" ]
