import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        if isinstance(message, bytes):
            await websocket.send(message)
            print("Binary message received: {0} bytes".format(len(message)))
        else:
            await websocket.send(f"Received text: {message}")


#start_server = websockets.serve(echo, "localhost", 9001)

#asyncio.run(echo, "localhost", 9001)
#asyncio.get_event_loop().run_until_complete(start_server)
#asyncio.get_event_loop().run_until_complete(echo())
#asyncio.get_event_loop().run_until_complete(websockets.serve(echo, "localhost", 9001))

#print("Server started on ws://localhost:9001")
#asyncio.get_event_loop().run_forever()