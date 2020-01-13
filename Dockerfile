ARG NAMESPACE
ARG TARGET_REF
FROM ${NAMESPACE}/base_python3.6:${TARGET_REF}

LABEL maintainer="matthieu.simonin@inria.fr"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates unzip


## Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
RUN mkdir -p /opt/prog
WORKDIR /opt/prog


COPY . .
COPY docker/. .

RUN conda env create -f SESAME_PY3CPU.yml

RUN cd chkpt && unzip *.zip
RUN cd data && unzip *.zip
## Activate the env by default
RUN conda init bash

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate SESAME_PY3CPU" >> ~/.bashrc

ENTRYPOINT ["./main"]