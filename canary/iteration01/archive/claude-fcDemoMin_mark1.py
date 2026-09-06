import IPython.display as ipd
import numpy as np
import librosa
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR

# load the model
pretrained_model = "stt_en_conformer_transducer_large"
asr_model = EncDecRNNTBPEModel.from_pretrained(pretrained_model)

# get some model metadata
model_cfg = asr_model.cfg
if hasattr(asr_model, 'tokenizer'):
    tokenizer = asr_model.tokenizer
else:
    tokenizer = asr_model.decoder.tokenizer

# set up the model for streaming
cfg = OmegaConf.create(
    dict(
        frame_len=1.0,
        total_buffer=4.0,
        chunk_len=0.4,
    )
)

frame_asr = FrameBatchASR(
    asr_model=asr_model, frame_len=cfg.frame_len, 
    total_buffer=cfg.total_buffer, batch_size=1
)

# change 1: buffer constants and settings
RATE = 16000
CHUNK = int(cfg.chunk_len * RATE)  # 16000 * 0.4 = 6400 samples
BUFFER_SIZE = CHUNK * 10

def process_audio_from_buffer(client_id, client_queues):
    """Process audio from a theoretical buffer client_queues[client_id]["incoming_audio"]"""
   
    # Initialize for processing
    frame_asr.reset()
    frame_asr.read_audio_file_buffer = np.array([], dtype=np.float32)
    
    # Start streaming
    print("* Streaming audio from buffer...")
    
    # create placeholder for the transcriptions
    text = ""
    
    while True:
        # Check if there's audio data in the buffer
        if len(client_queues[client_id]["incoming_audio"]) > 0:
            # Get audio data from buffer
            audio_data = client_queues[client_id]["incoming_audio"].pop(0)
            
            # Convert audio data (16-bit PCM) to float32
            float_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Process the chunk
            frame_asr.read_audio_file_buffer = np.append(frame_asr.read_audio_file_buffer, float_data)
            
            # ASR inference
            frame_asr.run()
            
            # Get the current transcription
            transcription = frame_asr.get_hypotheses()[-1]
            
            # Only print if the transcription changed
            if transcription != text:
                print(transcription)
                text = transcription
                
            # If this is the end of the processing (you'll need to define this condition)
            # For example, could be a special message in the queue
            if len(client_queues[client_id].get("end_of_stream", False)):
                frame_asr.flush()
                transcription = frame_asr.get_hypotheses()[-1]
                print(f"Final transcription: {transcription}")
                break
        
        # Optional: Add a small sleep to prevent CPU overload
        # time.sleep(0.01)

# Example usage:
# client_queues = {
#     "client1": {
#         "incoming_audio": [],  # Buffer where raw audio will be placed
#         "end_of_stream": False  # Flag to indicate stream completion
#     }
# }
# process_audio_from_buffer("client1", client_queues)