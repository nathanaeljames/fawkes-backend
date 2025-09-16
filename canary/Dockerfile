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
    ffmpeg\
    tzdata

ENV TZ America/New_York
    
RUN pip3 install \
    #autobahn[twisted]\
    #pyttsx3\
    #speechRecognition\
    #pyaudio\
    #scipy\
    #soundfile\
    #ibm-watson \
    #ibm_cloud_sdk_core \
    websockets \
    pydub \
    piper-tts==1.2.0\
    duckdb\
    aiohttp
# Explicitly install a compatible NumPy version first
# Based on the error, nemo-toolkit requires numpy<2.0.0,>=1.22 # - numpy 1.26.4 is the latest 1.x version
#RUN pip3 install numpy==1.26.4
#RUN pip3 install numpy==2.1.3
#SWITCH OUT FOR GPU
#RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
#RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
#RUN pip3 install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
#RUN pip3 install nemo_toolkit[nlp]
#RUN pip3 install nemo_toolkit[asr]
#RUN pip3 install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"
# Need latest directly from github to run Canary-qwen
RUN pip3 install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git" sacrebleu
#RUN pip3 install nemo_toolkit[all]
#RUN pip3 install nemo_toolkit[asr,nlp]
#TURN ON FOR GPU
RUN pip3 install cuda-python>=12.3
RUN pip3 install coqui-tts


#not sure why necessary to copy server.py explicitly but not doing so results in deadlocked/unreadable versions
COPY server*.py ./
COPY . .

EXPOSE 9001

#CMD ["python3", "server.py"]