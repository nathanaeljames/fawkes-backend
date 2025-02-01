FROM ubuntu:22.04
# running stably on Jammy Jellyfish
# (newer versions will require running pip in venv)

WORKDIR /usr/local/whisper

USER root

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update\
    && apt-get upgrade\
    && apt-get install python3-pip\
    && pip3 install autobahn[twisted]

COPY server.py ./

EXPOSE 9001

CMD ["python3", "server.py"]