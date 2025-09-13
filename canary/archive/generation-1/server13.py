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
IBM_API_KEY = "IYBIxRJeINqwcjOAd0PuFYI6NLyH0qV8hqfh3ziNqtQf"
IBM_SERVICE_URL = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/30d589a2-77a6-4819-90f7-9a3090278b40"

# IBM Watson setup
authenticator = IAMAuthenticator(IBM_API_KEY)
stt = SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(IBM_SERVICE_URL)

# Initialize pyttsx3 for text-to-speech
#tts_engine = pyttsx3.init()
#tts_engine.setProperty('rate', 150)

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
                    asyncio.run_coroutine_threadsafe(debug_stream_tts_audio(response_text), main_loop)
                #save_audio_segment(response_text)

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
    
    try:
        audio_data = generate_speech(text)  # Generate speech audio
        if audio_data is None:
            print("❌ Failed to generate speech audio. Exiting function.")
            return "AUDIO_GENERATION_FAILED"
    except Exception as e:
        print(f"❌ Exception in generate_speech: {e}")
        return "EXCEPTION_IN_GENERATE_SPEECH"

    chunk_size = 1024
    print(f"Active clients: {len(active_websockets)}")
    print(f"Audio data length: {len(audio_data)} bytes")  

    if len(audio_data) == 0:
        print("❌ ERROR: Audio data is empty! Exiting function.")
        return "EMPTY_AUDIO"

    # REMOVE CLOSED CONNECTIONS
    active_websockets_copy = active_websockets.copy()
    for ws in active_websockets_copy:
        if ws.closed:
            print("⚠️ Removing closed WebSocket")
            active_websockets.remove(ws)

    if not active_websockets:
        print("❌ No active WebSockets available. Exiting function.")
        return "NO_ACTIVE_WEBSOCKETS"

    print("✅ Starting loop to send audio chunks...")

    try:
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            print(f"🟢 Sending chunk {i // chunk_size + 1}: {len(chunk)} bytes")

            if active_websockets:
                for ws in active_websockets:
                    if not ws.closed:
                        try:
                            await ws.send(chunk)
                            print(f"✅ Sent {len(chunk)} bytes to clients")
                        except Exception as e:
                            print(f"❌ WebSocket send error: {e}")
                            return "WEBSOCKET_SEND_ERROR"
            else:
                print("⚠️ No active clients to send messages to.")
                return "NO_ACTIVE_CLIENTS"
            
            await asyncio.sleep(0.05)  # Simulating real-time streaming

        if active_websockets:
            print("✅ Sending EOF")
            await asyncio.gather(*[ws.send(b"EOF") for ws in active_websockets])

    except Exception as e:
        print(f"❌ Exception in sending audio: {e}")
        return "EXCEPTION_IN_SENDING_AUDIO"

    print("✅ Finished streaming audio")
    return "SUCCESS"

# Call the function and explicitly print the return value
async def debug_stream_tts_audio(text):
    result = await stream_tts_audio(text)
    print(f"🔍 Function returned: {result}")    

def generate_speech(text, voice="en+f3", speed=150, pitch=50):
    """Generate speech using espeak-ng and return properly formatted WAV audio."""
    command = [
        'espeak-ng',
        '-v', voice,
        '-s', str(speed),
        '-p', str(pitch),
        '--stdout',
        text
    ]

    print("Running espeak-ng subprocess...")  # Debugging

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_data, error = process.communicate(timeout=5)  # Timeout ensures it doesn't hang

        print("espeak-ng process completed.")  # Debugging

        if process.returncode != 0:
            print(f"espeak-ng error: {error.decode().strip()}")
            return None  # Ensure failure returns None

        if not audio_data:
            print("espeak-ng returned empty audio data.")
            return None

        print(f"espeak-ng produced {len(audio_data)} bytes of raw audio.")  # Debugging

        # Convert raw PCM to WAV using pydub
        audio_segment = AudioSegment.from_raw(io.BytesIO(audio_data), sample_width=2, frame_rate=22050, channels=1)

        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        return buffer.getvalue()

    except subprocess.TimeoutExpired:
        print("espeak-ng process timed out.")
        return None

    except Exception as e:
        print(f"Error in generate_speech: {e}")
        return None

def save_audio_segment(audio_data):
    #audio_data = generate_speech(text)
    # Convert the raw PCM data to an AudioSegment
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
    audio_segment.export("output_tts_01.wav", format="wav")

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
