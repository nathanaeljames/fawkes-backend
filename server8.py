import asyncio
import websockets

# WebSocket server settings
HOST = "localhost"
PORT = 9001

async def handler(websocket):
    """Handles incoming WebSocket connections."""
    print("Client connected.")
    try:
        async for message in websocket:
            print(f"Received: {message}")
            #await websocket.send(f"Echo: {message}")  # Send response back
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    """Starts the WebSocket server."""
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()  # Keep server running

if __name__ == "__main__":
    try:
        asyncio.run(main())  # Proper event loop handling for Python 3.10+
    except KeyboardInterrupt:
        print("Server shutting down.")
