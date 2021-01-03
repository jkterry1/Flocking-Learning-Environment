FROM python:3.8

COPY requirements.txt /

RUN pip3 install -r /requirements.txt

WORKDIR /birdflocking

COPY . /birdflocking

RUN ./build.sh

CMD ["python", "test_flocking_api.py"]