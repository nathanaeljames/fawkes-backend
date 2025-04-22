import asyncio
import websockets
import json
import io
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# IBM Watson Speech-to-Text credentials
IBM_API_KEY = "IYBIxRJeINqwcjOAd0PuFYI6NLyH0qV8hqfh3ziNqtQf"
IBM_SERVICE_URL = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/30d589a2-77a6-4819-90f7-9a3090278b40"

# WebSocket server settings
HOST = "localhost"
PORT = 9001

# IBM Watson setup
authenticator = IAMAuthenticator(IBM_API_KEY)
stt = SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(IBM_SERVICE_URL)

async def process_audio(websocket):
    """Handles incoming L16 audio data and sends it to IBM Watson."""
    async for message in websocket:
        with io.BytesIO(message) as audio:
            try:
                watson_ws = stt.recognize_using_websocket(
                    audio=audio,
                    content_type="audio/l16; rate=16000",
                    recognize_callback=WatsonCallback(),
                    interim_results=True
                )
                async for message in websocket:
                    await watson_ws.send(message)
            except Exception as e:
                print(f"Error: {e}")
            finally:
                await websocket.close()
                print("Client disconnected")

class WatsonCallback:
    def on_transcription(self, transcript):
        print("Transcription:", transcript)

    def on_error(self, error):
        print("Error:", error)

async def main():
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    server = await websockets.serve(process_audio, HOST, PORT)
    await server.wait_closed()  # Keep the server running

# Ensure asyncio is properly managed
if __name__ == "__main__":
    try:
        asyncio.run(main())  # This correctly starts the event loop
    except KeyboardInterrupt:
        print("Server shutting down.")
