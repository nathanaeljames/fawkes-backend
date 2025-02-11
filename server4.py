import asyncio
import websockets
import json
import ssl
import base64
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# IBM Watson Speech-to-Text credentials
IBM_API_KEY = "IYBIxRJeINqwcjOAd0PuFYI6NLyH0qV8hqfh3ziNqtQf"
IBM_SERVICE_URL = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/30d589a2-77a6-4819-90f7-9a3090278b40"

# WebSocket server settings
HOST = "localhost"
PORT = 9001

authenticator = IAMAuthenticator(IBM_API_KEY)
stt = SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(IBM_SERVICE_URL)

async def process_audio(websocket):
    """Handles incoming L16 audio data and sends it to IBM Watson."""
    try:
        watson_ws = stt.recognize_using_websocket(
            audio=message,
            content_type="audio/l16; rate=16000",
            recognize_callback=WatsonCallback(),
            interim_results=True
        )
        async for message in websocket:
            await watson_ws.send(message)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await watson_ws.close()

class WatsonCallback:
    def on_transcription(self, transcript):
        print("Transcription:", transcript)

    def on_error(self, error):
        print("Error:", error)

async def main():
    server = await websockets.serve(process_audio, HOST, PORT)
    print(f"WebSocket server listening on ws://{HOST}:{PORT}")
    await server.wait_closed()  # Keeps the server running

asyncio.run(main())