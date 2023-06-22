FROM python:3.8-slim

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get -y install gnupg \
&& apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 0E98404D386FA1D9 F8D2585B8783D481 6ED0E7B82643E131 BDE6D2B9216EC7A8 648ACFD622F3D138 54404762BBB6E853 \
&& apt-get -y install libspatialindex-dev --no-install-recommends \
&& rm -rf /var/lib/apt/lists/* \
&& /usr/local/bin/python -m pip install --upgrade pip

COPY . .

RUN pip3 install --no-cache-dir --compile -e .

ENTRYPOINT ["elara"]
