import asyncio
import websockets #pip install websockets
#import speech_recognition as sr #pip install speechRecognition
import io
#import pyttsx3 #pip install pyttsx3
import wave
from pydub import AudioSegment
import datetime
import json
import subprocess
from ibm_watson import SpeechToTextV1, TextToSpeechV1
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
active_websockets = set()  # Store active clients

# Dialogue partners
SPEAKER = "Nathanael"
SERVER = "Fawkes"

# IBM Watson Speech-to-Text credentials
IBM_API_KEY_STT = "IYBIxRJeINqwcjOAd0PuFYI6NLyH0qV8hqfh3ziNqtQf"
IBM_API_KEY_TTS = "IYBIxRJeINqwcjOAd0PuFYI6NLyH0qV8hqfh3ziNqtQf"
IBM_SERVICE_URL_STT = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/30d589a2-77a6-4819-90f7-9a3090278b40"
IBM_SERVICE_URL_TTS = "https://api.us-south.text-to-speech.watson.cloud.ibm.com/instances/64729709-cd8e-450f-b0e1-7ab112763ac0"

# IBM Watson setup
#authenticator = IAMAuthenticator(IBM_API_KEY_STT)
stt = SpeechToTextV1(authenticator=IAMAuthenticator(IBM_API_KEY_STT))
stt.set_service_url(IBM_SERVICE_URL_STT)

# IBM Watson setup (TTS)
#authenticator = IAMAuthenticator(IBM_API_KEY_STT)
tts = TextToSpeechV1(authenticator=IAMAuthenticator(IBM_API_KEY_STT))
tts.set_service_url(IBM_SERVICE_URL_TTS)  # Use the TTS-specific service URL

# define callback for the speech to text service
class WatsonCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_transcription(self, transcript):
        #print("transcription called")
        #print(transcript)
        #if active_websockets:
        #    asyncio.run_coroutine_threadsafe(send_message_to_clients(SPEAKER + ': ' + str(transcript)), main_loop)
        #if 'the time' in str(transcript):
        #    print("Asked about the time")
        #    strTime = datetime.datetime.now().strftime("%H:%M:%S")    
            #speak(f"Sir, the time is {strTime}")
        #    if active_websockets:
        #        asyncio.run_coroutine_threadsafe(send_message_to_clients(SERVER + ': Sir, the time is ' + strTime), main_loop)
        pass

    def on_connected(self):
        print('Connection was successful')

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

    def on_listening(self):
        print('Service is listening')

    def on_hypothesis(self, hypothesis):
        #print("hypothesis called")
        #print(hypothesis)
        pass
        #asyncio.create_task(send_message_to_clients(str(hypothesis)))
        #loop = asyncio.get_running_loop()
        #loop.call_soon_threadsafe(asyncio.create_task, send_message_to_clients(str(hypothesis)))
        #if active_websockets:
        #    asyncio.run_coroutine_threadsafe(send_message_to_clients(SPEAKER + ': ' + str(hypothesis)), main_loop)

    def on_data(self, data):
        #print("on_data called")
        print(data)
        #json_string = '{"speaker": SPEAKER, "final": data.final, "transcript": "New York"}'
        transcript_text = data['results'][0]['alternatives'][0]['transcript']
        is_final = data['results'][0]['final']
        data_to_send = {
            "speaker": SPEAKER,
            "final": is_final,
            "transcript": transcript_text
        }
        json_string = json.dumps(data_to_send)
        if active_websockets:
            asyncio.run_coroutine_threadsafe(send_message_to_clients(json_string), main_loop)        
        #pass
        if(is_final):
            print("Current speaker is done speaking")
            # here is where to house all response routines
            if 'the time' in transcript_text.lower():
                print("Asked about the time")
                strTime = datetime.datetime.now().strftime("%H:%M:%S")
                response_text = f"Sir, the time is {strTime}"
                data_to_send = {
                    "speaker": SERVER,
                    "final": "True",
                    "transcript": response_text
                }
                json_string = json.dumps(data_to_send)  
                #speak(f"Sir, the time is {strTime}")
                if active_websockets:
                    asyncio.run_coroutine_threadsafe(send_message_to_clients(json_string), main_loop)
                # Send response as TTS audio
                if active_websockets:
                    asyncio.run_coroutine_threadsafe(stream_tts_audio(response_text), main_loop)

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

async def receive_audio_service(websocket):
    """Handles incoming WebSocket connections."""
    print("Client connected.")
    active_websockets.add(websocket)  # Store the connection
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                #print("Binary message received: {0} bytes".format(len(message)))
                #await websocket.send(message)
                #audio_frames.append(message)  # Store raw PCM data
                try:
                    q.put(message)
                    #print("Received audio data and added to queue")
                except Full:
                    print("WARNING: packets dropped!")
                    pass # discard
            else:
                print(f"Text message received: {message}")
                #await websocket.send(f"Received text: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        active_websockets.remove(websocket)  # Remove connection when done
        # Save the collected L16 audio data as a WAV file
        #save_as_wav(b''.join(audio_frames), "output.wav")

async def transcribe_audio_service():
    """Initiates IBM Watson transcription service."""
    recognize_thread = Thread(target=recognize_using_weboscket, args=())
    recognize_thread.start()

async def send_message_to_clients(message):
    """Send a message to all connected WebSocket clients."""
    if active_websockets:
        await asyncio.gather(*[ws.send(message) for ws in active_websockets])
    else:
        print("No active clients to send messages to.")

# Stream IBM Watson TTS audio back to WebSocket clients
async def stream_tts_audio(text):
    """Converts text to speech and streams it over WebSockets."""
    print(f"TTS Response: {text}")

    response = tts.synthesize(
        text,
        voice="en-US_AllisonV3Voice",
        accept="audio/wav"
    ).get_result()

    for chunk in response.iter_content(1024):
        if active_websockets:
            await asyncio.gather(*[ws.send(chunk) for ws in active_websockets])
        await asyncio.sleep(0.1)  # Simulate real-time streaming

    # Send "EOF" signal to indicate the end of the audio stream
    if active_websockets:
        await asyncio.gather(*[ws.send("EOF") for ws in active_websockets])

async def main():
    global main_loop
    main_loop = asyncio.get_event_loop()  # Store the event loop
    # Start the WebSocket server for receiving audio
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    server = await websockets.serve(receive_audio_service, HOST, PORT)
    # Start the transcription process
    #await transcribe_audio_service()
    transcribe_task = asyncio.create_task(transcribe_audio_service())
    # Start saving the audio in a separate task
    #save_task = asyncio.create_task(save_audio())
    # Keep the server running
    await server.wait_closed()
    # Stop the saving process
    #await q.put(None)
    #await save_task
    await transcribe_task

if __name__ == "__main__":
    try:
        asyncio.run(main())  # Proper event loop handling for Python 3.10+
    except KeyboardInterrupt:
        # stop recording
        audio_source.completed_recording()  
        print("Server shutting down.")
