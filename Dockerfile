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
    #scipy\
    #soundfile\
    websockets \
    pydub \
    ibm-watson \
    ibm_cloud_sdk_core \
    piper-tts
RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
#RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
#RUN pip3 install nemo_toolkit[nlp]
#RUN pip3 install nemo_toolkit[asr]
RUN pip3 install nemo_toolkit[all]
#RUN pip3 install nemo_toolkit[asr,nlp]
#RUN pip3 install TTS
# Need to downgrade transformers to work with current usage/ model of Coqui
# In recent versions of transformers, the generation_config.pad_token_id is expected to be a GenerationConfig object — but XTTS passes it as a raw int
# Must be installed after nemo_toolkit[nlp] which requires transformers >= 4.41.0, but should be able to remove nemo_toolkit in final production
#RUN pip3 install transformers==4.35.2
RUN pip3 install cuda-python>=12.3
RUN pip3 install coqui-tts


#not sure why necessary to copy server.py explicitly but not doing so results in deadlocked/unreadable versions
COPY server*.py README.md ./
COPY . .

EXPOSE 9001

#CMD ["python3", "server.py"]