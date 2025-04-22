import asyncio
import websockets
import json
import queue
import ibm_watson
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Initialize in-memory queue
audio_queue = queue.Queue()

# IBM Watson Speech-to-Text credentials
API_KEY = 'your_ibm_api_key'
URL = 'your_ibm_url'

# Create Watson client
authenticator = IAMAuthenticator(API_KEY)
speech_to_text = SpeechToTextV1(authenticator=authenticator)
speech_to_text.set_service_url(URL)

async def transcribe_audio():
    """
    Function that listens to the queue and sends audio chunks to Watson's Speech to Text.
    """
    # Create a WebSocket connection to Watson's speech-to-text service
    websocket = await websockets.connect("wss://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/YOUR_INSTANCE_URL")

    # Sending start message to Watson's Speech-to-Text WebSocket
    start_message = {
        "action": "start",
        "content-type": "audio/l16; rate=16000; channels=1",
        "interim_results": True,
        "inactivity_timeout": -1  # Disable inactivity timeout
    }
    await websocket.send(json.dumps(start_message))

    # Listen to the queue and send audio chunks to Watson
    while True:
        if not audio_queue.empty():
            audio_chunk = audio_queue.get()
            await websocket.send(audio_chunk)
            print("Sent audio chunk to Watson")

        response = await websocket.recv()
        print("Received from Watson:", response)

    # Close the WebSocket connection
    await websocket.close()

async def receive_audio(websocket, path):
    """
    Function that receives audio data over WebSocket and places it in the queue.
    """
    while True:
        audio_data = await websocket.recv()
        audio_queue.put(audio_data)
        print("Received audio data and added to queue")

async def main():
    """
    Main function to set up WebSocket server and Watson transcription.
    """
    # Start the WebSocket server for receiving audio
    server = await websockets.serve(receive_audio, "localhost", 8765)

    # Start the transcription process
    await transcribe_audio()

    # Keep the server running
    await server.wait_closed()

if __name__ == "__main__":
    # Run the asyncio event loop
    asyncio.run(main())
