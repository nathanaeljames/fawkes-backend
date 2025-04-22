## @server.py
#  This file contains the server side of the Autobahn websocket service
import speech_recognition as sr #pip install speechRecognition
import datetime
import io
import pyaudio

from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from threading import Thread
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

try:
    from Queue import Queue, Full
except ImportError:
    from queue import Queue, Full

from twisted.internet import reactor

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory

class StreamingServerProtocol(WebSocketServerProtocol):

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        if isBinary:
            #audio = pyaudio.PyAudio()
            #stream = audio.open(
            #    format=pyaudio.paInt16,
            #    channels=1,
            #    rate=16000,
            #    input=True,
            #    frames_per_buffer=CHUNK,
            #    stream_callback=stream_callback,
            #    start=False
            #)
            stream = io.BytesIO(payload)
            print("Enter CTRL+C to end recording...")
            #stream.start_stream()

            try:
                recognize_thread = Thread(target=recognize_using_weboscket, args=())
                recognize_thread.start()

                while True:
                    pass
            except KeyboardInterrupt:
                # stop recording
                stream.stop_stream()
                stream.close()
                payload.terminate()
                audio_source.completed_recording()
        else:
            print("Text message received: {0}".format(payload.decode('utf8')))

        # echo back message verbatim
        #self.sendMessage(payload, isBinary)

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

###############################################
#### Initalize queue to store the recordings ##
###############################################
CHUNK = 1024
# Note: It will discard if the websocket client can't consumme fast enough
# So, increase the max size as per your choice
BUF_MAX_SIZE = CHUNK * 10
# Buffer to store audio
q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK)))

# Create an instance of AudioSource
audio_source = AudioSource(q, True, True)

###############################################
#### Prepare Speech to Text Service ########
###############################################

# initialize speech to text service
authenticator = IAMAuthenticator('IYBIxRJeINqwcjOAd0PuFYI6NLyH0qV8hqfh3ziNqtQf')
speech_to_text = SpeechToTextV1(authenticator=authenticator)

# define callback for the speech to text service
class MyRecognizeCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_transcription(self, transcript):
        print(transcript)

    def on_connected(self):
        print('Connection was successful')

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

    def on_listening(self):
        print('Service is listening')

    def on_hypothesis(self, hypothesis):
        print(hypothesis)

    def on_data(self, data):
        print(data)

    def on_close(self):
        print("Connection closed")

# this function will initiate the recognize service and pass in the AudioSource
def recognize_using_weboscket(*args):
    mycallback = MyRecognizeCallback()
    speech_to_text.recognize_using_websocket(audio=stream,
                                             content_type='audio/l16; rate=16000',
                                             recognize_callback=mycallback,
                                             interim_results=True)
    
def stream_callback(payload, frame_count, time_info, status):
    try:
        q.put(payload)
    except Full:
        pass # discard
    return (None)

if __name__ == '__main__':
   factory = WebSocketServerFactory("ws://localhost:9001")
   factory.protocol = StreamingServerProtocol
   reactor.listenTCP(9001, factory)
   reactor.run()
