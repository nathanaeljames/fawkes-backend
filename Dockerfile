FROM ubuntu:22.04
# running stably on Jammy Jellyfish
# (newer versions will require running pip in venv)

WORKDIR /usr/local/whisper

USER root

#update upgrade install must be on same line or will fail due to caching
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y\
    python3\
    python3-pip\
    nano\
    git
    
RUN pip3 install\
    autobahn[twisted]\
    pyttsx3\
    speechRecognition\
    ibm-watson\
    ibm_cloud_sdk_core\
    pyaudio

#not sure why necessary to copy server.py explicitly but not doing so results in deadlocked/unreadable version
COPY server.py ./
COPY . .

EXPOSE 9001

#CMD ["python3", "server.py"]