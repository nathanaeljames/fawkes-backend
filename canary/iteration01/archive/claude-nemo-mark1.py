# NeMo ASR Integration for WebSocket Server

import torch
import numpy as np
import asyncio
import json
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR

# Global variables for pre-loaded model
global_asr_model = None

def load_asr_model():
    """Load the ASR model once at server startup"""
    global global_asr_model
    
    print("Pre-loading NeMo ASR model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "/root/fawkes/models/fc-hybrid-lg-multi/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
    
    global_asr_model = EncDecRNNTBPEModel.restore_from(MODEL_PATH, map_location=torch.device(device))
    # Configure the model for streaming
    global_asr_model.encoder.set_default_att_context_size([70, 16])
    global_asr_model.change_decoding_strategy(decoder_type='rnnt')
    
    print("NeMo ASR model loaded successfully")
    return global_asr_model

class AsyncStreamingASR:
    """
    Asynchronous wrapper for NeMo's streaming ASR that works with your websocket server.
    This class handles continuous processing of audio from client_queues and outputs
    transcription results to the outgoing text queue.
    """
    def __init__(self, 
                 client_id, 
                 client_queues, 
                 model=None, 
                 frame_len=6400,  # 0.4s chunks at 16kHz
                 total_buffer=25600,  # 1.6s buffer total
                 speaker_name="SPEAKER",
                 server_name="SERVER"):
        """
        Initialize the streaming ASR service for a specific client.
        
        Args:
            client_id: Unique identifier for the client
            client_queues: Dict of queues for this client
            model: NeMo ASR model (if None, will use the global pre-loaded model)
            frame_len: Length of audio frames to process
            total_buffer: Size of the buffer to maintain
            speaker_name: Name to use for the speaker in transcription output
            server_name: Name to use for the server in transcription output
        """
        self.client_id = client_id
        self.client_queues = client_queues
        self.speaker_name = speaker_name
        self.server_name = server_name
        self.frame_len = frame_len
        self.total_buffer = total_buffer
        
        # Use the pre-loaded model if available
        global global_asr_model
        if model is not None:
            self.asr_model = model
        elif global_asr_model is not None:
            self.asr_model = global_asr_model
        else:
            # Fallback to loading the model if neither is available
            print(f"[ASR] No pre-loaded model found for client {client_id}, loading model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            MODEL_PATH = "/root/fawkes/models/fc-hybrid-lg-multi/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
            self.asr_model = EncDecRNNTBPEModel.restore_from(MODEL_PATH, map_location=torch.device(device))
            self.asr_model.encoder.set_default_att_context_size([70, 16])
            self.asr_model.change_decoding_strategy(decoder_type='rnnt')
            
        # Create the frame batch processor
        self.frame_batch_asr = FrameBatchASR(
            asr_model=self.asr_model,
            frame_len=frame_len,
            total_buffer=total_buffer
        )
        
        # Tracking variables
        self.last_transcript = ""
        self.silence_frames = 0
        self.max_silence_frames = 15  # Adjust based on your needs (~1.5 seconds of silence)
        self.running = False
        self.task = None
        
    async def start(self):
        """Start the ASR processing loop as an asyncio task"""
        if self.running:
            return
            
        self.running = True
        self.task = asyncio.create_task(self._processing_loop())
        return self.task
        
    async def stop(self):
        """Stop the ASR processing loop"""
        if not self.running:
            return
            
        self.running = False
        if self.task:
            await self.task
            
    async def _processing_loop(self):
        """Main processing loop that handles audio from the queue"""
        print(f"[ASR] Starting streaming ASR for client {self.client_id}")
        
        # Reset the ASR for this session
        self.frame_batch_asr.reset()
        
        # Buffer to accumulate audio chunks
        audio_buffer = np.array([], dtype=np.float32)
        
        try:
            while self.running and self.client_id in self.client_queues:
                # Check if there's audio to process
                try:
                    # Get audio data with a short timeout to keep the loop responsive
                    chunk = await asyncio.wait_for(
                        self.client_queues[self.client_id]["incoming_audio"].get(), 
                        timeout=0.1
                    )
                    
                    # Check for None marker which indicates end of stream
                    if chunk is None:
                        print(f"[ASR] End of stream marker received for client {self.client_id}")
                        # Process any remaining audio
                        if len(audio_buffer) > 0:
                            transcriptions = self.frame_batch_asr.transcribe(audio_buffer)
                            if transcriptions and len(transcriptions) > 0:
                                await self._send_transcription(transcriptions[-1], is_final=True)
                        break
                    
                    # Convert from bytes to the format needed by NeMo
                    if isinstance(chunk, bytes):
                        # Convert from 16-bit PCM to float32 normalized to [-1, 1]
                        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Add to buffer
                        audio_buffer = np.concatenate((audio_buffer, audio_np))
                        
                        # Check if we have enough audio to process
                        while len(audio_buffer) >= self.frame_len:
                            # Extract frame for processing
                            frame = audio_buffer[:self.frame_len]
                            audio_buffer = audio_buffer[self.frame_len:]
                            
                            # Check if this is a silent frame (for end-of-utterance detection)
                            is_silent = np.abs(frame).mean() < 0.01  # Adjust threshold as needed
                            
                            # Process the audio frame
                            with torch.no_grad():
                                transcriptions = self.frame_batch_asr.transcribe(frame)
                                
                                if transcriptions and len(transcriptions) > 0:
                                    current_text = transcriptions[-1].strip()
                                    
                                    # Determine if this should be considered a final result
                                    is_final = False
                                    
                                    # If we have text and then hit silence, mark as final
                                    if is_silent and self.last_transcript:
                                        self.silence_frames += 1
                                        if self.silence_frames >= self.max_silence_frames:
                                            is_final = True
                                            self.silence_frames = 0
                                    else:
                                        self.silence_frames = 0
                                    
                                    # If text changed, send an update
                                    if current_text and current_text != self.last_transcript:
                                        self.last_transcript = current_text
                                        await self._send_transcription(current_text, is_final)
                                    
                                    # If final, reset tracking
                                    if is_final:
                                        self.last_transcript = ""
                
                except asyncio.TimeoutError:
                    # No audio data available, just continue
                    pass
                except Exception as e:
                    print(f"[ASR] Error processing audio for client {self.client_id}: {e}")
                
        except asyncio.CancelledError:
            print(f"[ASR] ASR task cancelled for client {self.client_id}")
        except Exception as e:
            print(f"[ASR] Unexpected error in ASR loop for client {self.client_id}: {e}")
        finally:
            # Clean up
            print(f"[ASR] Stopping streaming ASR for client {self.client_id}")
    
    async def _send_transcription(self, text, is_final=False):
        """Send transcription results to the outgoing text queue"""
        if not text:
            return
            
        print(f"[ASR] Client {self.client_id} - {'Final' if is_final else 'Interim'}: {text}")
        
        # Create the transcript JSON in the same format as your Watson callback
        data_to_send = {
            "speaker": self.speaker_name,
            "final": is_final,
            "transcript": text
        }
        json_string = json.dumps(data_to_send)
        
        # Send to the outgoing queue
        await self.client_queues[self.client_id]["outgoing_text"].put(json_string)
        
        # Handle final transcriptions that need responses
        if is_final:
            await self._handle_final_transcription(text)
    
    async def _handle_final_transcription(self, text):
        """Handle final transcriptions with the same response logic as your Watson callback"""
        # This function mimics the behavior in your Watson callback
        # You can extend this with more response types as needed
        
        # Check for time request
        if 'the time' in text.lower():
            print("[ASR] Asked about the time")
            import datetime
            str_time = datetime.datetime.now().strftime("%H:%M:%S")
            response_text = f"Sir, the time is {str_time}"
            
            data_to_send = {
                "speaker": self.server_name,
                "final": True,
                "transcript": response_text
            }
            json_string = json.dumps(data_to_send)
            
            # Send response text
            await self.client_queues[self.client_id]["outgoing_text"].put(json_string)
            
            # Signal to generate TTS (your existing code handles this)
            # You may want to modify this to fit your specific architecture
            return {"generate_tts": True, "text": response_text}
            
        # Check for name request
        elif 'your name' in text.lower():
            print("[ASR] Asked about my name")
            response_text = "My name is Neil Richard Gaiman."
            
            data_to_send = {
                "speaker": self.server_name,
                "final": True,
                "transcript": response_text
            }
            json_string = json.dumps(data_to_send)
            
            # Send response text
            await self.client_queues[self.client_id]["outgoing_text"].put(json_string)
            
            # Signal to generate xtts (your existing code handles this)
            return {"generate_xtts": True, "speaker": "neil_gaiman", "text": response_text}
            
        return None

# Function to integrate with your websocket server
async def run_nemo_streaming_asr(client_id, client_queues, speaker_name="SPEAKER", server_name="SERVER", model=None):
    """
    Run the NeMo streaming ASR for a client and handle results
    
    This function is designed to be run as an asyncio task, integrating with
    your existing websocket handling code.
    """
    print(f"[NeMo ASR] Starting ASR service for client {client_id}")
    
    # Create the ASR processor for this client
    asr_processor = AsyncStreamingASR(
        client_id=client_id,
        client_queues=client_queues,
        speaker_name=speaker_name,
        server_name=server_name,
        model=model
    )
    
    # Start the processor and wait for it to complete
    try:
        await asr_processor.start()
    except Exception as e:
        print(f"[NeMo ASR] Error in ASR service for client {client_id}: {e}")
    finally:
        # Ensure cleanup
        await asr_processor.stop()
        print(f"[NeMo ASR] ASR service stopped for client {client_id}")

# Usage in your main function:
'''
async def main():
    global main_loop
    main_loop = asyncio.get_event_loop()
    
    # Pre-load the ASR model at server startup
    asr_model = load_asr_model()
    
    # Other initialization code
    load_speakers_into_manager()
    
    # Start the WebSocket server
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    server = await websockets.serve(
        lambda ws: connection_handler(ws, asr_model),  # Pass model to connection handler
        HOST, PORT
    )
    
    # Start and keep the server running
    await server.wait_closed()

# Update your connection handler
async def connection_handler(websocket, asr_model):
    client_id = str(uuid.uuid4())
    print(f"New client connected: {client_id}")
    await websocket_server(websocket, client_id, asr_model)

# Update your websocket_server function
async def websocket_server(websocket, client_id, asr_model):
    active_websockets[client_id] = websocket
    client_queues[client_id] = {
        "incoming_audio": asyncio.Queue(),
        "outgoing_audio": asyncio.Queue(),
        "outgoing_text": asyncio.Queue(),
    }
    
    try:
        incoming_task = asyncio.create_task(handle_incoming(websocket, client_id))
        outgoing_task = asyncio.create_task(handle_outgoing(websocket, client_id))
        
        # Use NeMo ASR instead of Watson
        asr_task = asyncio.create_task(
            run_nemo_streaming_asr(
                client_id=client_id,
                client_queues=client_queues,
                speaker_name=SPEAKER,
                server_name=SERVER,
                model=asr_model  # Pass the pre-loaded model
            )
        )
        
        await asyncio.gather(incoming_task, outgoing_task, asr_task)
    except asyncio.CancelledError:
        print(f"WebSocket task for {client_id} cancelled.")
    except Exception as e:
        print(f"WebSocket error for {client_id}: {e}")
    finally:
        print(f"Cleaning up client {client_id}")
        if client_id in client_queues:
            client_queues.pop(client_id)
        if client_id in active_websockets:
            active_websockets.pop(client_id)
        await websocket.close()
'''