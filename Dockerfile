FROM ubuntu:22.04
# running stably on Jammy Jellyfish
# (newer versions will require running pip in venv)

WORKDIR /usr/local/whisper

USER root

#update upgrade install must be on same line or will fail due to caching
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y\
    #portaudio19-dev\
    #espeak-ng\
    python3\
    python3-pip\
    nano\
    git\
    ffmpeg
    
RUN pip3 install \
    #autobahn[twisted]\
    #pyttsx3\
    #speechRecognition\
    #pyaudio\
    websockets \
    pydub \
    ibm-watson \
    ibm_cloud_sdk_core \
    piper-tts
RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
#RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install TTS


#not sure why necessary to copy server.py explicitly but not doing so results in deadlocked/unreadable versions
COPY server*.py README.md ./
COPY . .

EXPOSE 9001

#CMD ["python3", "server.py"]