## @server.py
#  This file contains the server side of the Autobahn websocket service
import speech_recognition as sr #pip install speechRecognition
import datetime
import io

from twisted.internet import reactor

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory

class StreamingServerProtocol(WebSocketServerProtocol):

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        if isBinary:
            r = sr.Recognizer()
            #data_buffer = io.BytesIO(payload)
            #print("Binary message received: {0} bytes".format(len(payload)))
            print("Listening...")
            #r.pause_threshold = 1
            #audio = r.listen(data_buffer)
            #try:
            #   print("Recognizing...")    
            #   query = r.recognize_google(data_buffer, language='en-in')
            #   print(f"User said: {query}\n")

            #except Exception as e:
            #   print(e)    
            #   print("Say that again please...")

            try:
                with io.BytesIO(payload) as audio_file:
                    with sr.AudioFile(audio_file) as source:
                        audio_data = r.record(source)
                
                # Perform speech recognition
                try:
                    query = r.recognize_google(audio_data)
                    #print(f"Recognized text: {text}")
                    print(f"User said: {query}\n")
                    # Send the recognized text back to the client or process it further
                    #await websocket.send(text)

                except sr.UnknownValueError:
                    print("Speech recognition could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results from speech recognition service; {e}")
            
            except Exception as e:
                    print(f"Error processing audio data: {e}") 
        else:
            print("Text message received: {0}".format(payload.decode('utf8')))

        # echo back message verbatim
        #self.sendMessage(payload, isBinary)

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

if __name__ == '__main__':
   factory = WebSocketServerFactory("ws://localhost:9001")
   factory.protocol = StreamingServerProtocol
   reactor.listenTCP(9001, factory)
   reactor.run()
