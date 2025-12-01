FROM nvidia/cuda:12.3.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://security.ubuntu.com/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list


RUN apt-get update --fix-missing && \
    apt-get install -y \
        python3 python3-pip python3-venv \
        git curl build-essential \
        && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace

