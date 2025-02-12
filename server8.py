import asyncio
import websockets #pip install websockets
import speech_recognition as sr #pip install speechRecognition
import io
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from threading import Thread
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

try:
    from Queue import Queue, Full
except ImportError:
    from queue import Queue, Full

# WebSocket server settings
HOST = "localhost"
PORT = 9001

# IBM Watson Speech-to-Text credentials
IBM_API_KEY = "IYBIxRJeINqwcjOAd0PuFYI6NLyH0qV8hqfh3ziNqtQf"
IBM_SERVICE_URL = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/30d589a2-77a6-4819-90f7-9a3090278b40"

# IBM Watson setup
authenticator = IAMAuthenticator(IBM_API_KEY)
stt = SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(IBM_SERVICE_URL)

# define callback for the speech to text service
class WatsonCallback(RecognizeCallback):
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

# this function will initiate the recognize service and pass in the AudioSource
def recognize_using_weboscket(*args):
    stt.recognize_using_websocket(audio=audio_source,
                                content_type='audio/l16; rate=16000',
                                recognize_callback=WatsonCallback(),
                                interim_results=True)

async def receive_audio(websocket):
    """Handles incoming WebSocket connections."""
    print("Client connected.")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                #print("Binary message received: {0} bytes".format(len(message)))
                #await websocket.send(message)
                try:
                    q.put(message)
                    #print("Received audio data and added to queue")
                except Full:
                    pass # discard
            else:
                print(f"Text message received: {message}")
                #await websocket.send(f"Received text: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")

async def transcribe_audio():
    recognize_thread = Thread(target=recognize_using_weboscket, args=())
    recognize_thread.start()  

async def main():
    """Starts the WebSocket server."""
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    #async with websockets.serve(receive_audio, HOST, PORT):
    #    await asyncio.Future()  # Keep server running

    # Start the WebSocket server for receiving audio
    server = await websockets.serve(receive_audio, HOST, PORT)
    # Start the transcription process
    await transcribe_audio()
    # Keep the server running
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())  # Proper event loop handling for Python 3.10+
    except KeyboardInterrupt:
        # stop recording
        audio_source.completed_recording()  
        print("Server shutting down.")
