FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

COPY requirements.txt /

RUN apt update
RUN apt install -y python3-pip
RUN apt install -y git

RUN DEBIAN_FRONTEND="noninteractive" apt install -y python3-opencv

RUN pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r /requirements.txt

WORKDIR /birdflocking

COPY . /birdflocking

RUN ./build.sh

CMD ["python3", "train.py"]