import asyncio
import websockets
import json
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback
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

class WatsonCallback(RecognizeCallback):
    """Handles IBM Watson responses."""
    
    def on_transcription(self, transcript):
        print("Transcription:", transcript)

    def on_error(self, error):
        print("Error:", error)

async def process_audio(websocket):
    """Receives audio from WebSocket and forwards it to IBM Watson."""
    print("Client connected")

    # Prepare Watson WebSocket connection
    callback = WatsonCallback()

    # Create WebSocket connection to IBM Watson
    with open("temp_audio.l16", "wb") as audio_file:
        try:
            async for message in websocket:
                audio_file.write(message)  # Save incoming audio data

            # Re-open file in binary read mode and send to Watson
            with open("temp_audio.l16", "rb") as audio:
                stt.recognize_using_websocket(
                    audio=audio,
                    content_type="audio/l16; rate=16000",
                    recognize_callback=callback,
                    interim_results=True
                )

        except Exception as e:
            print(f"Error: {e}")
        finally:
            await websocket.close()
            print("Client disconnected")

async def main():
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    
    server = await websockets.serve(process_audio, HOST, PORT)
    await server.wait_closed()  # Keep server running

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutting down.")
