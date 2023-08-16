# ONEAPI on Nvidia GPU 
This project is to enable docker execution using oneAPI library to perform execution on NVIDIA GPU.

## Docker Build
To build docker on the file run the execution as below
```
docker build --network=host -f Dockerfile -t oneapi-cuda:cuda-11.7.1-oneapi-2023.0.0 . 
```

or you can download the docker image from the dockerhub below
```
docker pull raymondpyn/oneapi-cuda:cuda-11.7.1-oneapi-2023.0.0
```

## Docker run 
To run the docker the file simply execute the command below 

- **repo host dir**: Host directory which contain this repository
- **container dir**: Container directory which mapped from host directory
```
docker run -it --privileged -v <repo host dir>:<container dir> --network=host --gpus all  raymondpyn/oneapi-cuda:cuda-11.7.1-oneapi-2023.0.0 bash
```
Run environment setup to support NVIDIA GPU
```
bash /opt/oneapi-for-nvidia-gpus-2023.0.0-linux.sh && \
    . /opt/intel/oneapi/setvars.sh --include-intel-llvm
```
Go to the directory, run cmake and make run
```
cd <container dir>
cmake .
make run
```

