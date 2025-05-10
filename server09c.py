# hooking into rasa
import asyncio
from asyncio import Queue
import websockets #pip install websockets
#import speech_recognition as sr #pip install speechRecognition
import io
#import pyttsx3 #pip install pyttsx3
import wave
from pydub import AudioSegment
import datetime
import json
#import subprocess
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from threading import Thread
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from piper.voice import PiperVoice #pip install piper-tts
import torch
#import torchaudio
#from coqui_tts.models import XTTS
#from xtts import XTTS
#from TTS.api import TTS # Coqui XTTS API
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
#from TTS.api import TTS
import numpy as np
#import soundfile as sf
from pathlib import Path
#from collections import defaultdict
#import struct
import uuid
import audioop
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models import EncDecRNNTBPEModel
#from nemo.collections.asr.parts.streaming.frame_batch_asr import FrameBatchASR
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
import copy
import time
import atexit  # To handle exits
import struct  # For writing binary data to WAV

# WebSocket server settings
#HOST = "localhost"
HOST = "0.0.0.0"
PORT = 9001
#active_websockets = set()  # Store active clients
active_websockets = {}  # client_id -> websocket
client_queues = {}
clientSideTTS = False

# Dialogue partners
SPEAKER = "Nathanael"
SERVER = "Fawkes"

# Establish the preferred device to run models on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if (DEVICE != "cuda"):
    print('Warning! GPU not detected!')

# Load Piper TTS with northern_english_male (med) voice
model_path = "/root/fawkes/models/piper_tts/en_GB-northern_english_male-medium.onnx"
pipervoice = PiperVoice.load(model_path)

# Load the Coqui XTTS model
MODEL_DIR = Path("/root/fawkes/models/coqui_xtts/XTTS-v2/")
CONFIG_PATH = MODEL_DIR / "config.json"
SPEAKERS_DIR = Path("speakers")
# Load model
xtts_config = XttsConfig()
xtts_config.load_json(CONFIG_PATH)
xtts_model = Xtts.init_from_config(xtts_config)
xtts_model.load_checkpoint(
    config=xtts_config,
    checkpoint_dir=MODEL_DIR,
    eval=True
)
xtts_model.to(DEVICE)

# Load cache-aware STT model stt_en_fastconformer_hybrid_large_streaming_multi
# --- NeMo ASR Configuration ---
NEMO_MODEL_PATH = "/root/fawkes/models/fc-hybrid-lg-multi/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
NEMO_ENCODER_STEP_LENGTH = 80  # ms (for FastConformer)
NEMO_LOOKAHEAD_SIZE = 480  # 0ms, 80ms, 480ms, 1040ms lookahead / 80ms, 160ms, 540ms, 1120ms chunk size
NEMO_DECODER_TYPE = 'rnnt'
NEMO_SAMPLE_RATE = 16000 # Hz

class NemoStreamingTranscriber:
    def __init__(self, model_path, decoder_type, lookahead_size, encoder_step_length, device, sample_rate):
        self.device = device
        self.sample_rate = sample_rate
        self.encoder_step_length = encoder_step_length
        self.model_path = model_path
        self.decoder_type = decoder_type
        self.lookahead_size = lookahead_size
        self.asr_model = self._load_model()
        self.preprocessor = self._init_preprocessor()
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = self.asr_model.encoder.get_initial_cache_state(
            batch_size=1)
        self.previous_hypotheses = None
        self.pred_out_stream = None
        self.step_num = 0
        self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
        num_channels = self.asr_model.cfg.preprocessor.features
        self.cache_pre_encode = torch.zeros((1, num_channels, self.pre_encode_cache_size),
                                           device=self.device)

    def _load_model(self):
        print("Pre-loading NeMo ASR model...")
        asr_model = EncDecRNNTBPEModel.restore_from(self.model_path, map_location=torch.device(self.device))
        asr_model.eval()
        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = False
            if hasattr(asr_model, 'joint'):
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
        asr_model.change_decoding_strategy(decoding_cfg)
        if "multi" in self.model_path:
            left_context_size = asr_model.encoder.att_context_size[0]
            asr_model.encoder.set_default_att_context_size(
                [left_context_size, int(self.lookahead_size / self.encoder_step_length)])
        print("NeMo ASR model loaded successfully")
        return asr_model

    def _init_preprocessor(self):
        cfg = copy.deepcopy(self.asr_model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        preprocessor.to(self.device)
        return preprocessor

    def _preprocess_audio(self, audio):
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(self.device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(self.device)
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        return processed_signal, processed_signal_length

    def _extract_transcriptions(self, hyps):
        if isinstance(hyps[0], Hypothesis):
            transcriptions = [hyp.text for hyp in hyps]
        else:
            transcriptions = hyps
        return transcriptions

    def transcribe_chunk(self, new_chunk):
        audio_data = new_chunk.astype(np.float32) / 32768.0
        processed_signal, processed_signal_length = self._preprocess_audio(audio_data)
        processed_signal = torch.cat([self.cache_pre_encode, processed_signal], dim=-1)
        processed_signal_length += self.cache_pre_encode.shape[1]
        self.cache_pre_encode = processed_signal[:, :, -self.pre_encode_cache_size:]
        with torch.no_grad():
            (
                self.pred_out_stream,
                transcribed_texts,
                self.cache_last_channel,
                self.cache_last_time,
                self.cache_last_channel_len,
                self.previous_hypotheses,
            ) = self.asr_model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=self.previous_hypotheses,
                previous_pred_out=self.pred_out_stream,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )
        final_streaming_tran = self._extract_transcriptions(transcribed_texts)
        self.step_num += 1
        return final_streaming_tran[0]

async def websocket_server(websocket, client_id, nemo_transcriber):
    active_websockets[client_id] = websocket
    client_queues[client_id] = {
        "incoming_audio": asyncio.Queue(),
        "outgoing_audio": asyncio.Queue(),
        "outgoing_text": asyncio.Queue(),
        "raw_asr": asyncio.Queue(),
    }
    try:
        incoming_task = asyncio.create_task(handle_incoming(websocket, client_id))
        outgoing_task = asyncio.create_task(handle_outgoing(websocket, client_id))
        #asr_task = asyncio.create_task(run_streaming_asr(client_id))
        #audio_recorder = AudioRecorder()
        asr_task = asyncio.create_task(process_audio_from_queue(client_id, nemo_transcriber))
        await asyncio.gather(incoming_task, outgoing_task, asr_task)
    except asyncio.CancelledError:
        print(f"WebSocket task for {client_id} cancelled.")
    except Exception as e:
        print(f"WebSocket error for {client_id}: {e}")
    finally:
        print(f"Cleaning up client {client_id}")
        if client_id in client_queues:
            client_queues.pop(client_id)
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        await websocket.close()

async def handle_incoming(websocket, client_id):
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                await client_queues[client_id]["incoming_audio"].put(message)
            else:
                print(f"Text message received: {message}")
                if(message == 'clientSideTTS'):
                    global clientSideTTS
                    clientSideTTS = True
                    print(f"Client has specified using client-side TTS.")
    except websockets.exceptions.ConnectionClosed:
        print(f"Client {client_id} disconnected.")
    finally:
        active_websockets.pop(client_id, None)
        client_queues.pop(client_id, None)

async def handle_outgoing(websocket, client_id):
    try:
        while True:
            # Check both audio and text queues continuously
            if client_id not in client_queues:
                print(f"Client {client_id} no longer in client_queues. Exiting handle_outgoing.")
                break
            if not client_queues[client_id]["outgoing_audio"].empty():
                chunk = await client_queues[client_id]["outgoing_audio"].get()
                if chunk is None:
                    #print(f"Sending chunk: {chunk!r}")
                    await websocket.send(b"EOF")
                    continue
                else:
                    #print(f"Sending chunk: {chunk!r}")
                    await websocket.send(chunk)
            elif not client_queues[client_id]["outgoing_text"].empty():
                text = await client_queues[client_id]["outgoing_text"].get()
                #print(f"Outgoing text in queue is {text}.")
                await websocket.send(text)
            else:
                await asyncio.sleep(0.01)  # avoid busy loop
    except websockets.exceptions.ConnectionClosed:
        pass

def prepare_for_streaming(chunk, type, rate):
    """Convert a chunk of audio into 16kHz 16-bit mono PCM for streaming."""
    TARGET_RATE = 16000
    TARGET_WIDTH = 2  # 16-bit
    TARGET_CHANNELS = 1  # mono
    if type == "wav":
        # Convert WAV bytes to AudioSegment
        audio = AudioSegment.from_file(io.BytesIO(chunk), format="wav")
        # Convert to desired format
        audio = audio.set_frame_rate(TARGET_RATE).set_channels(TARGET_CHANNELS).set_sample_width(TARGET_WIDTH)
        return audio.raw_data
    elif type == "raw":
        # Input is already PCM, just resample if needed
        if rate != TARGET_RATE:
            chunk = audioop.ratecv(chunk, TARGET_WIDTH, TARGET_CHANNELS, rate, TARGET_RATE, None)[0]
        return chunk
    elif type == "float32":
        # Assume chunk is a NumPy float32 array in range [-1.0, 1.0]
        if isinstance(chunk, np.ndarray) and chunk.dtype == np.float32:
            # Resample to 16kHz if needed
            if rate != TARGET_RATE:
                # Use linear interpolation resampling
                import librosa
                chunk = librosa.resample(chunk, orig_sr=rate, target_sr=TARGET_RATE)
            # Convert float32 [-1.0, 1.0] to int16 PCM
            chunk = np.clip(chunk, -1.0, 1.0)
            chunk_int16 = (chunk * 32767).astype(np.int16)
            return chunk_int16.tobytes()
        else:
            raise ValueError("Expected NumPy float32 array for type='float32'")
    else:
        raise ValueError(f"Unsupported audio type: {type}")

async def send_message_to_clients(client_id, message):
    """Send a message to all connected WebSocket clients."""
    #print("Send message to clients called.")
    client_queues[client_id]["outgoing_text"].put_nowait(message)
    #if active_websockets:
    #    await asyncio.gather(*[ws.send(message) for ws in active_websockets])
    #else:
    #    print("No active clients to send messages to.")

async def stream_tts_audio(client_id, text):
    print(f"[TTS] Streaming (raw) to client {client_id}: {text}")
    
    if client_id not in client_queues:
        print(f"[TTS] Client {client_id} not in client_queues")
        return

    try:
        for chunk in pipervoice.synthesize_stream_raw(text):
            # Use dict to indicate raw PCM
            converted_chunk = await asyncio.to_thread(prepare_for_streaming, chunk, 'raw', pipervoice.config.sample_rate)
            #await client_queues[client_id]["outgoing_audio"].put({"type": "raw_pcm", "data": chunk, "rate": pipervoice.config.sample_rate})
            await client_queues[client_id]["outgoing_audio"].put(converted_chunk)
            await asyncio.sleep(0.015)

        await client_queues[client_id]["outgoing_audio"].put(None)

    except Exception as e:
        print(f"[TTS] Error streaming to {client_id}: {e}")
    
def load_speakers_into_manager():
    """
    Load all .pt files in the speakers directory into speaker_manager.
    """
    for pt_file in SPEAKERS_DIR.glob("*.pt"):
        data = torch.load(pt_file, map_location="cpu")
        xtts_model.speaker_manager.speakers[pt_file.stem] = {
            "gpt_cond_latent": data["gpt_cond_latent"],
            "speaker_embedding": data["speaker_embedding"]
        }
    print(f"Loaded {len(xtts_model.speaker_manager.speakers)} speakers into speaker_manager.")

async def synthesize_xtts_audio(speaker_name, text, buffer):
    print(f"Computing XTTS for: {text}")
    speaker_data = xtts_model.speaker_manager.speakers.get(speaker_name)
    if speaker_data is None:
        raise ValueError("Speaker not found.")
    #assert isinstance(speaker_data["gpt_cond_latent"], torch.Tensor), "gpt_cond_latent is not a Tensor"
    #assert isinstance(speaker_data["speaker_embedding"], torch.Tensor), "speaker_embedding is not a Tensor"

    def blocking_inference():
        #print(f"[XTTS] Executing blocking loop.")
        try:
            chunks = xtts_model.inference_stream(
                text=text,
                language="en",
                gpt_cond_latent=speaker_data["gpt_cond_latent"].to(xtts_model.device),
                speaker_embedding=speaker_data["speaker_embedding"].to(xtts_model.device),
                stream_chunk_size=512,
            )
            for chunk in chunks:
                if chunk is None or len(chunk) == 0:
                    continue
                # Move from GPU to CPU
                chunk_np = chunk.cpu().numpy()
                # Convert to streamable format (in thread)
                converted_chunk = prepare_for_streaming(chunk_np, 'float32', 22050)
                # Send it to the buffer queue from this thread
                asyncio.run_coroutine_threadsafe(
                    buffer.put(converted_chunk),
                    main_loop
                )
            # Signal end-of-stream
            asyncio.run_coroutine_threadsafe(
                buffer.put(None),
                main_loop
            )
            print(f"[XTTS] Finished streaming to buffer for client {client_id}")
        except Exception as e:
            print(f"[XTTS] Error during inference: {e}")
            asyncio.run_coroutine_threadsafe(buffer.put(None), main_loop)

    # Run the blocking work in a background thread
    await asyncio.to_thread(blocking_inference)

async def stream_xtts_audio(client_id, buffer):
    print(f"[XTTS] Streaming from buffer queue to client {client_id}")
    queue = client_queues.get(client_id, {}).get("outgoing_audio")
    if not queue:
        print(f"[XTTS] Client {client_id} not in client_queues or missing 'outgoing_audio'")
        return
    try:
        while True:
            chunk = await buffer.get()
            #print(f"[XTTS] Got chunk from buffer: {chunk}")
            if chunk is None:  # End-of-stream marker
                break
            await queue.put(chunk)
            await asyncio.sleep(0.015)
        await queue.put(None)
    except Exception as e:
        print(f"[XTTS] Error streaming to {client_id}: {e}")

async def synthesize_stream_xtts_audio(client_id, speaker_name, text):
    """Generates speech using Coqui XTTS and streams it over WebSockets."""
    print(f"Streaming XTTS for: {text}")
    speaker_data = xtts_model.speaker_manager.speakers.get(speaker_name)

    if speaker_data is None:
        raise ValueError(f"Speaker '{speaker_name}' not found in speaker_manager.")

    chunks = xtts_model.inference_stream(
        text=text,
        language="en",
        gpt_cond_latent=speaker_data["gpt_cond_latent"].to(xtts_model.device),
        speaker_embedding=speaker_data["speaker_embedding"].to(xtts_model.device),
        stream_chunk_size=512,  # optional, tune for latency/quality
    )

    try:
        for chunk in chunks:
            # chunk is a NumPy float32 array at 24kHz
            if chunk is None or len(chunk) == 0:
                continue
            chunk_np = chunk.cpu().numpy()
            converted_chunk = await asyncio.to_thread(prepare_for_streaming, chunk_np, 'float32', 22050)
            await client_queues[client_id]["outgoing_audio"].put(converted_chunk)
            await asyncio.sleep(0.015)
        await client_queues[client_id]["outgoing_audio"].put(None)
    except Exception as e:
        print(f"[XTTS] Error streaming to {client_id}: {e}")

async def process_audio_from_queue(client_id, nemo_transcriber):
    """
    Processes audio chunks from an asyncio.Queue.
    """
    chunk_size_ms = NEMO_LOOKAHEAD_SIZE + NEMO_ENCODER_STEP_LENGTH
    bytes_per_chunk = int(NEMO_SAMPLE_RATE * chunk_size_ms / 1000) * 2  # 2 bytes per sample (int16)
    audio_buffer = b''  # Initialize an empty byte buffer
    previous_transcriptions = []
    stability_threshold = 4  # (5) Number of chunks to consider for stability
    #silence_threshold_ms = 300  # (750) Silence threshold in milliseconds
    #last_speech_time = time.time()  # Initialize last speech time
    previous_text = ""  # Store the previously sent text 

    try:
        while True:
            try:
                #chunk_count = 0
                #audio_chunk = await client_queues[client_id]["incoming_audio"].get()  # Use await
                #text = transcribe_chunk(audio_chunk)
                audio_data = await client_queues[client_id]["incoming_audio"].get()
                if audio_data is None or len(audio_data) == 0:
                    await asyncio.sleep(0.001)  # Or handle empty chunk as appropriate
                    continue  # Skip processing empty chunk
                audio_buffer += audio_data
                while len(audio_buffer) >= bytes_per_chunk:
                    chunk_bytes = audio_buffer[:bytes_per_chunk]
                    audio_buffer = audio_buffer[bytes_per_chunk:]
                    audio_chunk = np.frombuffer(chunk_bytes, dtype=np.int16)
                    if audio_chunk.ndim != 1:
                        audio_chunk = audio_chunk.squeeze()
                    if audio_chunk.dtype != np.int16:
                        audio_chunk = audio_chunk.astype(np.int16)
                    text = await asyncio.to_thread(nemo_transcriber.transcribe_chunk, audio_chunk)
                    #print(text)
                    # --- Finality Detection Logic ---
                    # 1. Update previous transcriptions
                    previous_transcriptions.append(text)
                    if len(previous_transcriptions) > stability_threshold:
                        previous_transcriptions.pop(0)  # Keep only the last N transcriptions
                    # 2. Check for stability
                    is_stable = False
                    if len(previous_transcriptions) == stability_threshold:
                        is_stable = all(t == previous_transcriptions[0] for t in previous_transcriptions)
                    # 3. Check for silence (crude approximation)
                    #is_silent = len(audio_chunk) < 200  # (100) Adjust this threshold
                    #print(f"chunk_len: {len(audio_chunk)}, is_silent: {is_silent}")
                    #print("is silent")
                    #current_time = time.time()
                    #time_since_last_speech = current_time - last_speech_time
                    #print("time since last speech = " + time_since_last_speech)
                    # Hybrid Finality Decision
                    #if is_stable:
                    #    print("transcription stability event")
                    #if is_silent and time_since_last_speech > silence_threshold_ms / 1000:
                    #    print("silence event")
                    #is_final = is_stable or (is_silent and time_since_last_speech > silence_threshold_ms / 1000)
                    is_final = is_stable
                    # Update last speech time if we detect speech
                    #if not is_silent:
                    #    last_speech_time = current_time
                    #    print(f"Speech detected, last_speech_time updated to: {last_speech_time}")
                    new_text = text[len(previous_text):].strip()  # Extract the new part
                    # Send data (either final or interim)
                    if new_text:
                        data_to_send = {
                            "speaker": SPEAKER,
                            "final": is_final,
                            "transcript": new_text  # Send only the new part
                        }
                        json_string = json.dumps(data_to_send)
                        await send_message_to_clients(client_id, json_string)
                    if is_final:
                        previous_transcriptions = []  # Reset for the next utterance
                        previous_text = text  # Update previous_text

            except asyncio.QueueEmpty:  # asyncio uses asyncio.QueueEmpty
                await asyncio.sleep(0.01)  # Use asyncio.sleep
            except Exception as e:
                print(f"Error processing audio: {e}")
                break
            finally:
                client_queues[client_id]["incoming_audio"].task_done() # Necessary for asyncio.Queue

    finally:
        print("Async Audio processing stopped")

async def connection_handler(websocket):
    global client_id, nemo_transcriber
    client_id = str(uuid.uuid4())
    print(f"New client connected: {client_id}")
    await websocket_server(websocket, client_id, nemo_transcriber)

async def main():
    global main_loop,nemo_transcriber
    main_loop = asyncio.get_event_loop()  # Store the event loop
    #load all speakers one time into dictionary
    load_speakers_into_manager()
        # Initialize the Nemo transcriber
    nemo_transcriber = NemoStreamingTranscriber(
        model_path=NEMO_MODEL_PATH,
        decoder_type=NEMO_DECODER_TYPE,
        lookahead_size=NEMO_LOOKAHEAD_SIZE,
        encoder_step_length=NEMO_ENCODER_STEP_LENGTH,
        device=DEVICE,
        sample_rate=NEMO_SAMPLE_RATE
    )
    # Start the WebSocket server for receiving audio
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    server = await websockets.serve(connection_handler, HOST, PORT)
    #transcribe_task = asyncio.create_task(transcribe_audio_service())
    # Start and keep the server running
    await server.wait_closed()
    # Start the transcription process
    #await transcribe_task

if __name__ == "__main__":
    try:
        asyncio.run(main())  # Proper event loop handling for Python 3.10+
    except KeyboardInterrupt:
        # stop recording
        #audio_source.completed_recording()  
        print("Server shutting down.")
