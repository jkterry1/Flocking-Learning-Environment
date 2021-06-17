FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

COPY requirements.txt /

RUN pip3 install -r /requirements.txt

WORKDIR /birdflocking

COPY . /birdflocking

RUN ./build.sh

CMD ["python", "test_flocking_api.py"]