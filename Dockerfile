FROM thinkwhere/gdal-python:3.6-ubuntu

COPY . .
RUN pip3 install -e .

ENTRYPOINT ["elara"]
