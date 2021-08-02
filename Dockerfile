#!/bin/sh
FROM python:3.8

COPY requirements.txt /

RUN pip3 install -r /requirements.txt

RUN pip3 install supersuit --no-deps

WORKDIR /birdflocking

COPY . /birdflocking

RUN ./build.sh

CMD ["python", "test-wrappers.py"]
