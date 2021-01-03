from python:3.8

WORKDIR /birdflocking

COPY . /birdflocking

RUN pip3 install -r requirements.txt

RUN ./build.sh

CMD ["python", "test_flocking_api.py"]