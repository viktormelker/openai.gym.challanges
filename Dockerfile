
FROM gw000/keras:2.1.4-py3

# install swig to be able to use box2d-py package
RUN apt-get update -qq \
    && apt-get install --no-install-recommends -y \
    # install essentials
    swig \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/prod.txt /tmp/
RUN pip3 install --process-dependency-links --no-cache-dir -r /tmp/requirements.txt

WORKDIR /openai
COPY . /app

# default command
# CMD ["/app/run.py"]