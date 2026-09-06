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

AUDIO_FILE_PATH = "output2.wav"

async def stream_audio(websocket):
    """Stream an audio file in chunks to the WebSocket client."""
    print("Client connected.")

    chunk_size = 1024  # Send 1024-byte chunks

    while True:
        try:
            with open(AUDIO_FILE_PATH, "rb") as audio_file:
                while chunk := audio_file.read(chunk_size):
                    print(f"Sending chunk of size {len(chunk)} bytes")
                    await websocket.send(chunk)  # Send binary data
                    await asyncio.sleep(0.05)  # Simulate real-time streaming

            print("Finished streaming. Sending EOF")
            await websocket.send(b"EOF")  # Send EOF signal

        except Exception as e:
            print(f"Error streaming audio: {e}")

        print("Client disconnected.")

async def main():
    global main_loop
    main_loop = asyncio.get_event_loop()  # Store the event loop
    # Start the WebSocket server for receiving audio
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    #server = await websockets.serve(receive_audio_service, HOST, PORT)
    server = await websockets.serve(stream_audio, HOST, PORT)
    # Start the transcription process
    #await transcribe_audio_service()
    #transcribe_task = asyncio.create_task(transcribe_audio_service())
    # Start saving the audio in a separate task
    #save_task = asyncio.create_task(save_audio())
    # Keep the server running
    await server.wait_closed()
    # Stop the saving process
    #await q.put(None)
    #await save_task
    #await transcribe_task

if __name__ == "__main__":
    try:
        asyncio.run(main())  # Proper event loop handling for Python 3.10+
    except KeyboardInterrupt:
        # stop recording
        #audio_source.completed_recording()  
        print("Server shutting down.")
