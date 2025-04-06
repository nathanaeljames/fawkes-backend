# This server replaces Watson TTS with piper and adds Coqui for zero-shot voice cloning
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
from piper.voice import PiperVoice #pip install piper-tts
import torch
#import torchaudio
#from coqui_tts.models import XTTS
#from xtts import XTTS
#from TTS.api import TTS # Coqui XTTS API
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
#from TTS.api import TTS
import numpy as np
import soundfile as sf

try:
    from Queue import Queue, Full
except ImportError:
    from queue import Queue, Full

# WebSocket server settings
#HOST = "localhost"
HOST = "0.0.0.0"
PORT = 9001
active_websockets = set()  # Store active clients
clientSideTTS = False

# Dialogue partners
SPEAKER = "Nathanael"
SERVER = "Fawkes"

# IBM Watson Speech-to-Text (STT) Credentials
IBM_STT_API_KEY = "IYBIxRJeINqwcjOAd0PuFYI6NLyH0qV8hqfh3ziNqtQf"
IBM_STT_SERVICE_URL = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/30d589a2-77a6-4819-90f7-9a3090278b40"
# Initialize IBM Watson STT
stt_authenticator = IAMAuthenticator(IBM_STT_API_KEY)
stt = SpeechToTextV1(authenticator=stt_authenticator)
stt.set_service_url(IBM_STT_SERVICE_URL)

# Load Piper TTS with northern_english_male (med) voice
model_path = "/root/fawkes/models/piper_tts/en_GB-northern_english_male-medium.onnx"
pipervoice = PiperVoice.load(model_path)

# Load the Coqui XTTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
xtts_config = XttsConfig()
xtts_config.load_json("/root/fawkes/models/coqui_xtts/XTTS-v2/config.json")
xtts_model = Xtts.init_from_config(xtts_config)
xtts_model.load_checkpoint(
    config=xtts_config,
    checkpoint_dir="/root/fawkes/models/coqui_xtts/XTTS-v2/",
    eval=True
)
xtts_model.to(device)

# define callback for the speech to text service
class WatsonCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_transcription(self, transcript):
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
        pass

    def on_data(self, data):
        print(data)
        # Establish transcript JSON
        transcript_text = data['results'][0]['alternatives'][0]['transcript']
        is_final = data['results'][0]['final']
        data_to_send = {
            "speaker": SPEAKER,
            "final": is_final,
            "transcript": transcript_text
        }
        json_string = json.dumps(data_to_send)
        # Send transcript as text/ JSON
        if active_websockets:
            asyncio.run_coroutine_threadsafe(send_message_to_clients(json_string), main_loop)        
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
                # Send response as text/ JSON
                if active_websockets:
                    asyncio.run_coroutine_threadsafe(send_message_to_clients(json_string), main_loop)
                # Send response as TTS audio
                if not clientSideTTS and active_websockets:
                    main_loop.call_soon_threadsafe(asyncio.create_task, stream_tts_audio(response_text))
            if 'your name' in transcript_text.lower():
                print("Asked about my name")
                #response_text = f"Sir, my name is {SERVER}"
                response_text = "My name is Neil Richard Esteban Gaiman mother fucker!"
                data_to_send = {
                    "speaker": SERVER,
                    "final": "True",
                    "transcript": response_text
                }
                json_string = json.dumps(data_to_send)
                # Send response as text/ JSON
                if active_websockets:
                    asyncio.run_coroutine_threadsafe(send_message_to_clients(json_string), main_loop)
                # Send response as TTS audio
                if not clientSideTTS and active_websockets:
                    #main_loop.call_soon_threadsafe(asyncio.create_task, stream_tts_audio(response_text))
                    main_loop.call_soon_threadsafe(asyncio.create_task, stream_xtts_audio(response_text,'/root/fawkes/audio_samples/neilgaiman_01.wav'))

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
def recognize_using_websocket(*args):
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
                try:
                    q.put(message)
                    #print("Received audio data and added to queue")
                except Full:
                    print("WARNING: packets dropped!")
                    pass # discard
            else:
                print(f"Text message received: {message}")
                if(message == 'clientSideTTS'):
                    global clientSideTTS
                    clientSideTTS = True
                    print(f"Client has specified using client-side TTS.")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        active_websockets.remove(websocket)  # Remove connection when done

async def transcribe_audio_service():
    """Initiates IBM Watson transcription service."""
    recognize_thread = Thread(target=recognize_using_websocket, args=())
    recognize_thread.start()

async def send_message_to_clients(message):
    """Send a message to all connected WebSocket clients."""
    if active_websockets:
        await asyncio.gather(*[ws.send(message) for ws in active_websockets])
    else:
        print("No active clients to send messages to.")

async def stream_tts_audio(text):
    """Streams generated TTS audio to connected WebSocket clients."""
    print(f"Streaming TTS for: {text}")
    
    # Create an in-memory buffer
    audio_stream = io.BytesIO()
    # Generate speech using Piper (writing directly to a wave file in memory)
    with wave.open(audio_stream, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(pipervoice.config.sample_rate)  # Use Piper's native sample rate
        pipervoice.synthesize(text, wav_file)
    # Convert Piper output to raw PCM at 16kHz
    audio_stream.seek(0)
    audio = AudioSegment.from_wav(audio_stream)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16kHz, mono, 16-bit PCM
    # Get raw PCM data
    audio_data = audio.raw_data

    if audio_data is None:
        print("Failed to generate speech audio. Exiting function.")
        return

    chunk_size = 1024
    print(f"Audio data length: {len(audio_data)} bytes")

    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        #print(f"Sending chunk {i // chunk_size + 1}: {len(chunk)} bytes")

        if active_websockets:
            await asyncio.gather(*[ws.send(chunk) for ws in active_websockets if ws.close_code is None])
        else:
            print("No active clients to send messages to.")
        
        await asyncio.sleep(0.015)

    if active_websockets:
        print("Sending EOF")
        await asyncio.gather(*[ws.send(b"EOF") for ws in active_websockets])

#def extract_voice_embedding(audio_path, save_path):
#    """Extracts a voice embedding from an audio file and saves it."""
#    embedding = xtts_model.get_speaker_embedding(audio_path)
#    np.save(save_path, embedding)

async def stream_xtts_audio(text, audio_sample_path):
    """Generates speech using Coqui XTTS and streams it over WebSockets."""
    print(f"Streaming XTTS for: {text}")

    # Load the stored speaker embedding from the .npy file
    #speaker_embedding = np.load(embedding_path)
    # Synthesize speech using XTTS (outputs 24kHz audio)
    #audio_wav = xtts_model.tts(text=text, speaker_wav=audio_sample_path, language="en")
    #audio_wav = xtts_model.tts_with_vc(text=text, speaker_wav=audio_sample_path, language="en")
    audio_wav_dict = xtts_model.synthesize(
        text=text,
        config=xtts_config,
        speaker_wav=audio_sample_path,
        language="en"
    )

    audio_wav_np = audio_wav_dict["wav"]  # NumPy float32 array
    sample_rate = 24000  # XTTS default

    # Write NumPy array to BytesIO as WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio_wav_np, samplerate=sample_rate, format="WAV")
    buffer.seek(0)

    # Convert to 16kHz mono for streaming
    audio = AudioSegment.from_wav(buffer)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio_data = audio.raw_data

    if not audio_data:
        print("Failed to generate speech audio. Exiting function.")
        return
    
    chunk_size = 1024
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        if active_websockets:
            await asyncio.gather(*[ws.send(chunk) for ws in active_websockets if ws.close_code is None])
        await asyncio.sleep(0.015)
    
    if active_websockets:
        print("Sending EOF")
        await asyncio.gather(*[ws.send(b"EOF") for ws in active_websockets])

async def main():
    global main_loop
    main_loop = asyncio.get_event_loop()  # Store the event loop
    # Start the WebSocket server for receiving audio
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    server = await websockets.serve(receive_audio_service, HOST, PORT)
    transcribe_task = asyncio.create_task(transcribe_audio_service())
    # Start and keep the server running
    await server.wait_closed()
    # Start the transcription process
    await transcribe_task

if __name__ == "__main__":
    try:
        asyncio.run(main())  # Proper event loop handling for Python 3.10+
    except KeyboardInterrupt:
        # stop recording
        audio_source.completed_recording()  
        print("Server shutting down.")
