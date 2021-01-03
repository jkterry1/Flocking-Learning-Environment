FROM python:3.8-alpine

RUN apk add \
    git \
    openblas-dev \
    gfortran \
    gcc \
    g++ 

COPY requirements.txt /

RUN pip3 install -r /requirements.txt

WORKDIR /birdflocking

COPY . /birdflocking

RUN ./build.sh

CMD ["python", "test_flocking_api.py"]