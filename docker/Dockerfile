# oneapi-cuda:cuda-11.7.1-oneapi-2023.0.0

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get clean && apt-get update && \
    apt-get install -y python3-dev git ninja-build cmake pkg-config python3-pip wget libssl-dev curl software-properties-common

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \ | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update 

RUN apt-get install -y intel-basekit=2023.0.0-25537 && \
apt-get install -y --allow-downgrades intel-oneapi-compiler-dpcpp-cpp=2023.0.0-25370

COPY oneapi-for-nvidia-gpus-2023.0.0-linux.sh /opt
RUN bash /opt/oneapi-for-nvidia-gpus-2023.0.0-linux.sh && \
    . /opt/intel/oneapi/setvars.sh --include-intel-llvm