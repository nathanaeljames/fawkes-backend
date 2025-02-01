FROM ubuntu:22.04
# running stably on Jammy Jellyfish
# (newer versions will require running pip in venv)

WORKDIR /usr/local/whisper

USER root

ARG DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone\
    && apt-get update\
    && apt-get upgrade -y\
    #&& apt-get install python3-pip -y\
    #&& pip3 install autobahn[twisted]\
    && apt-get install python3-pip -y\
    && apt-get install nano\
    #&& apt install pipx -y\
    #python3 -m pip install pipx\
    #&& pipx ensurepath\
    #&& pipx install autobahn[twisted] --python python3\
    && apt-get install python3-autobahn[twisted]

COPY server.py ./

EXPOSE 9001

CMD ["python3", "server.py"]