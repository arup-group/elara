FROM python:3.8-slim

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get -y install gdal-bin libgdal-dev g++ proj-bin libspatialindex-dev --no-install-recommends \
&& rm -rf /var/lib/apt/lists/* \
&& /usr/local/bin/python -m pip install --upgrade pip

COPY . .

RUN pip3 install --no-cache-dir --compile -e .

ENTRYPOINT ["elara"]
