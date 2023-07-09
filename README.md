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
```
docker run -it --gpus all raymondpyn/oneapi-cuda:cuda-11.7.1-oneapi-2023.0.0 bash
git clone https://github.com/raymondpee/oneapi-cuda
cd src/sycl-init
cmake
make run
```

