# third attempt at replacing Watson STT with Nemo
# first successfull attempt at using Nemo
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

try:
    from Queue import Queue, Full
except ImportError:
    from queue import Queue, Full

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

# IBM Watson Speech-to-Text (STT) Credentials
IBM_STT_API_KEY = "IYBIxRJeINqwcjOAd0PuFYI6NLyH0qV8hqfh3ziNqtQf"
IBM_STT_SERVICE_URL = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/30d589a2-77a6-4819-90f7-9a3090278b40"
# Initialize IBM Watson STT
stt_authenticator = IAMAuthenticator(IBM_STT_API_KEY)
stt = SpeechToTextV1(authenticator=stt_authenticator)
stt.set_service_url(IBM_STT_SERVICE_URL)

# Load Piper TTS with northern_english_male (med) voice
model_path = "/root/fawkes/models/piper_tts/en_GB-northern_english_male-medium.onnx"
pipervoice = PiperVoice.load(model_path)

# Load the Coqui XTTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
if (device != "cuda"):
    print('Warning! GPU not detected!')
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
xtts_model.to(device)

# Load cache-aware STT model stt_en_fastconformer_hybrid_large_streaming_multi
print("Pre-loading NeMo ASR model...")
MODEL_PATH = "/root/fawkes/models/fc-hybrid-lg-multi/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
asr_model = EncDecRNNTBPEModel.restore_from(MODEL_PATH, map_location=torch.device(device))
#asr_model.encoder.set_default_att_context_size([70, 16])
ENCODER_STEP_LENGTH = 80 # ms
lookahead_size = 80 # in milliseconds
left_context_size = asr_model.encoder.att_context_size[0]
asr_model.encoder.set_default_att_context_size([left_context_size, int(lookahead_size / ENCODER_STEP_LENGTH)])
asr_model.change_decoding_strategy(decoder_type='rnnt')
print("NeMo ASR model loaded successfully")

# make sure the model's decoding strategy is optimal
decoding_cfg = asr_model.cfg.decoding
with open_dict(decoding_cfg):
    # save time by doing greedy decoding and not trying to record the alignments
    decoding_cfg.strategy = "greedy"
    decoding_cfg.preserve_alignments = False
    if hasattr(asr_model, 'joint'):  # if an RNNT model
        # restrict max_symbols to make sure not stuck in infinite loop
        decoding_cfg.greedy.max_symbols = 10
        # sensible default parameter, but not necessary since batch size is 1
        decoding_cfg.fused_batch_size = -1
    asr_model.change_decoding_strategy(decoding_cfg)

# set model to eval mode
asr_model.eval()

# get parameters to use as the initial cache state
cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
    batch_size=1
)

# init params we will use for streaming
previous_hypotheses = None
pred_out_stream = None
step_num = 0
pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
# cache-aware models require some small section of the previous processed_signal to
# be fed in at each timestep - we initialize this to a tensor filled with zeros
# so that we will do zero-padding for the very first chunk(s)
num_channels = asr_model.cfg.preprocessor.features
cache_pre_encode = torch.zeros((1, num_channels, pre_encode_cache_size), device=asr_model.device)

# helper function for extracting transcriptions
def extract_transcriptions(hyps):
    """
        The transcribed_texts returned by CTC and RNNT models are different.
        This method would extract and return the text section of the hypothesis.
    """
    #print("Entering extract_transcriptions")
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    #print("Exiting extract_transcriptions")
    return transcriptions

# define functions to init audio preprocessor and to
# preprocess the audio (ie obtain the mel-spectrogram)
def init_preprocessor(asr_model):
    #print("Entering init_preprocessor")
    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0
    cfg.preprocessor.normalize = "None"
    
    preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
    preprocessor.to(asr_model.device)
    #print("Exiting init_preprocessor")
    return preprocessor

preprocessor = init_preprocessor(asr_model)

def preprocess_audio(audio, asr_model):
    #print("Entering preprocess_audio")  # Debugging
    #print(f"  Input audio shape: {audio.shape}, dtype: {audio.dtype}")  # Debugging
    device = asr_model.device

    # doing audio preprocessing
    audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
    audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
    #print(f"  audio_signal shape: {audio_signal.shape}, dtype: {audio_signal.dtype}")  # Debugging
    processed_signal, processed_signal_length = preprocessor(
        input_signal=audio_signal, length=audio_signal_len
    )
    #print(f"  processed_signal shape: {processed_signal.shape}, dtype: {processed_signal.dtype}")  # Debugging
    #print("Exiting preprocess_audio")  # Debugging
    return processed_signal, processed_signal_length

def transcribe_chunk(new_chunk):
    #print("Entering transcribe_chunk")  # Debugging
    #print(f"  Input new_chunk shape: {new_chunk.shape}, dtype: {new_chunk.dtype}")  # Debugging
    global cache_last_channel, cache_last_time, cache_last_channel_len
    global previous_hypotheses, pred_out_stream, step_num
    global cache_pre_encode
    
    # new_chunk is provided as np.int16, so we convert it to np.float32
    # as that is what our ASR models expect
    audio_data = new_chunk.astype(np.float32)
    audio_data = audio_data / 32768.0
    #print(f"  audio_data shape: {audio_data.shape}, dtype: {audio_data.dtype}")  # Debugging

    # get mel-spectrogram signal & length
    processed_signal, processed_signal_length = preprocess_audio(audio_data, asr_model)
     
    # prepend with cache_pre_encode
    processed_signal = torch.cat([cache_pre_encode, processed_signal], dim=-1)
    processed_signal_length += cache_pre_encode.shape[1]
    #print(f"  processed_signal (after cat) shape: {processed_signal.shape}, dtype: {processed_signal.dtype}")  # Debugging
    
    # save cache for next time
    cache_pre_encode = processed_signal[:, :, -pre_encode_cache_size:]
    
    with torch.no_grad():
        (
            pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = asr_model.conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=pred_out_stream,
            drop_extra_pre_encoded=None,
            return_transcription=True,
        )
    #print(f"  transcribed_texts: {transcribed_texts}")  # Debugging
    
    final_streaming_tran = extract_transcriptions(transcribed_texts)
    step_num += 1
    
    #print("Exiting transcribe_chunk")
    return final_streaming_tran[0]

# define callback for the speech to text service
class WatsonCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_transcription(self, transcript):
        pass

    def on_connected(self):
        print('Connection was successful')

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

    def on_listening(self):
        print('Service is listening')

    def on_hypothesis(self, hypothesis):
        pass

    def on_data(self, data):
        print(data)
        # Establish transcript JSON
        transcript_text = data['results'][0]['alternatives'][0]['transcript']
        is_final = data['results'][0]['final']
        data_to_send = {
            "speaker": SPEAKER,
            "final": is_final,
            "transcript": transcript_text
        }
        json_string = json.dumps(data_to_send)
        # Send transcript as text/ JSON
        #send_message_to_clients(json_string, client_id)
        #if active_websockets:
        asyncio.run_coroutine_threadsafe(send_message_to_clients(client_id, json_string), main_loop)
        #main_loop.call_soon_threadsafe(asyncio.create_task, stream_tts_audio(response_text, client_id))   
        if(is_final):
            print("Current speaker is done speaking")
            # here is where to house all response routines
            if 'the time' in transcript_text.lower():
                print("Asked about the time")
                strTime = datetime.datetime.now().strftime("%H:%M:%S")
                response_text = f"Sir, the time is {strTime}"
                data_to_send = {
                    "speaker": SERVER,
                    "final": "True",
                    "transcript": response_text
                }
                json_string = json.dumps(data_to_send)  
                # Send response as text/ JSON
                #if active_websockets:
                asyncio.run_coroutine_threadsafe(send_message_to_clients(client_id, json_string), main_loop)
                # Send response as TTS audio
                if not clientSideTTS and active_websockets:
                    main_loop.call_soon_threadsafe(asyncio.create_task, stream_tts_audio(client_id, response_text))
            if 'your name' in transcript_text.lower():
                print("Asked about my name")
                #response_text = f"Sir, my name is {SERVER}"
                response_text = "My name is Neil Richard Gaiman."
                data_to_send = {
                    "speaker": SERVER,
                    "final": "True",
                    "transcript": response_text
                }
                json_string = json.dumps(data_to_send)
                # Send response as text/ JSON
                if active_websockets:
                    asyncio.run_coroutine_threadsafe(send_message_to_clients(client_id, json_string), main_loop)
                # Send response as TTS audio
                #if not clientSideTTS and active_websockets:
                    #main_loop.call_soon_threadsafe(asyncio.create_task, stream_tts_audio(response_text))
                    #main_loop.call_soon_threadsafe(asyncio.create_task, stream_xtts_audio(response_text,'/root/fawkes/audio_samples/neilgaiman_01.wav'))
                #    main_loop.call_soon_threadsafe(asyncio.create_task, stream_tts_audio('Compiling response, please wait a moment...'))
                #    main_loop.call_soon_threadsafe(asyncio.create_task, stream_xtts_audio('neil_gaiman',response_text))

                #if not clientSideTTS and active_websockets:
                #    async def speak_response_sequentially():
                #        await stream_tts_audio("Please wait a moment...")
                #        await stream_xtts_audio("neil_gaiman", response_text)

                #    main_loop.call_soon_threadsafe(
                #        lambda: asyncio.create_task(speak_response_sequentially())
                #    )

                if not clientSideTTS and active_websockets:
                    #main_loop.call_soon_threadsafe(asyncio.create_task, stream_tts_audio(client_id,'Compiling response, please wait a moment...'))
                    #main_loop.call_soon_threadsafe(asyncio.create_task, synthesize_stream_xtts_audio(client_id,'neil_gaiman',response_text))
                    async def parallel_tts_pipeline():
                        # Preload speaker data etc. up front, but defer generator work
                        buffer = asyncio.Queue()
                        # Kick off Coqui as a background task — it starts immediately!
                        coqui_task = asyncio.create_task(synthesize_xtts_audio("neil_gaiman", response_text, buffer))
                        #coqui_task = asyncio.to_thread(
                        #    lambda: asyncio.run(synthesize_xtts_audio("neil_gaiman", response_text, buffer))
                        #)
                        # Do Piper TTS first (synchronous, placeholder speech)
                        await stream_tts_audio(client_id, "Please wait...")
                        # Once Piper is done, begin streaming the already-buffering Coqui output
                        await stream_xtts_audio(client_id, buffer)
                        # Optional: wait for Coqui task to fully complete if needed
                        await coqui_task

                    main_loop.call_soon_threadsafe(lambda: asyncio.create_task(parallel_tts_pipeline()))


    def on_close(self):
        print("Connection closed")

###############################################
#### Initalize queue to store the recordings ##
###############################################
CHUNK = 1024
# Note: It will discard if the websocket client can't consumme fast enough
# So, increase the max size as per your choice
BUF_MAX_SIZE = CHUNK * 10
# Buffer to store audio
q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK)))

# Create an instance of AudioSource
audio_source = AudioSource(q, True, True)

# this function will initiate the recognize service and pass in the AudioSource
def recognize_using_websocket(*args):
    stt.recognize_using_websocket(audio=audio_source,
                                content_type='audio/l16; rate=16000',
                                recognize_callback=WatsonCallback(),
                                interim_results=True)

async def websocket_server(websocket, client_id):
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
        audio_recorder = AudioRecorder()
        asr_task = asyncio.create_task(process_audio_from_queue(client_id, audio_recorder))
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
                '''
                try:
                    q.put(message)
                    #print("Received audio data and added to queue")
                except Full:
                    print("WARNING: packets dropped!")
                    pass # discard
                '''
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

async def transcribe_audio_service():
    """Initiates IBM Watson transcription service."""
    recognize_thread = Thread(target=recognize_using_websocket, args=())
    recognize_thread.start()

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

class AudioRecorder:
    def __init__(self, filename="sanity_check_2.wav", sample_rate=16000):
        self.filename = filename
        self.sample_rate = sample_rate
        self.data = bytearray()  # Use bytearray for efficient appending
        self.wf = None
        self._open_wav()
        atexit.register(self._close_wav)  # Ensure WAV is closed on exit

    def _open_wav(self):
        """Opens the WAV file for writing."""
        self.wf = wave.open(self.filename, 'wb')
        self.wf.setnchannels(1)  # Mono audio
        self.wf.setsampwidth(2)  # 2 bytes per sample (16-bit)
        self.wf.setframerate(self.sample_rate)

    def _close_wav(self):
        """Closes the WAV file."""
        if self.wf:
            try:
                self.wf.setnframes(len(self.data) // 2)  # Correct frame count
                self.wf.writeframes(self.data)
                self.wf.close()
                print(f"Saved audio to {self.filename}")
            except Exception as e:
                print(f"Error closing WAV file: {e}")

    def add_data(self, data):
        """Adds audio data to the WAV file."""
        try:
            self.data.extend(data)  # Append bytes directly
        except Exception as e:
            print(f"Error writing to WAV file: {e}")

async def process_audio_from_queue(client_id, audio_recorder):
    """
    Processes audio chunks from an asyncio.Queue.
    """
    global cache_last_channel, cache_last_time, cache_last_channel_len
    global previous_hypotheses, pred_out_stream, step_num
    global cache_pre_encode

    try:
        while True:
            try:
                #chunk_count = 0
                audio_chunk = await client_queues[client_id]["incoming_audio"].get()  # Use await
                #text = transcribe_chunk(audio_chunk)
                if audio_chunk is None or len(audio_chunk) == 0:
                    await asyncio.sleep(0.001)  # Or handle empty chunk as appropriate
                    continue  # Skip processing empty chunk
                if not isinstance(audio_chunk, np.ndarray):
                    audio_chunk = np.frombuffer(audio_chunk, dtype=np.int16)
                if audio_chunk.ndim != 1:
                    audio_chunk = audio_chunk.squeeze()  # Or reshape as needed
                if audio_chunk.dtype != np.int16:
                    audio_chunk = audio_chunk.astype(np.int16)
                #print(f"Chunk {chunk_count} shape: {audio_chunk.shape}, dtype: {audio_chunk.dtype}")
                #chunk_count += 1
                #audio_recorder.add_data(audio_chunk)
                #audio_recorder.add_data(audio_chunk.tobytes())
                text = await asyncio.to_thread(transcribe_chunk, audio_chunk)
                print(text, end='\r')
                #client_queues[client_id]["transcriptions"].put_nowait(text) # Assuming you have a transcriptions queue
                #await client_queues[client_id]["outgoing_text"].put(text)
                #client_queues[client_id]["outgoing_text"].put_nowait(text)
                is_final = False
                data_to_send = {
                    "speaker": SPEAKER,
                    "final": is_final,
                    "transcript": text
                }
                json_string = json.dumps(data_to_send)
                await send_message_to_clients(client_id, json_string)

            except asyncio.QueueEmpty:  # asyncio uses asyncio.QueueEmpty
                await asyncio.sleep(0.01)  # Use asyncio.sleep
            except Exception as e:
                print(f"Error processing audio: {e}")
                break
            finally:
                client_queues[client_id]["incoming_audio"].task_done() # Necessary for asyncio.Queue

    finally:
        print()
        print("Async Audio processing stopped")

async def connection_handler(websocket):
    global client_id
    client_id = str(uuid.uuid4())
    print(f"New client connected: {client_id}")
    await websocket_server(websocket, client_id)

async def main():
    global main_loop
    main_loop = asyncio.get_event_loop()  # Store the event loop
    #load all speakers one time into dictionary
    load_speakers_into_manager()
    # Start the WebSocket server for receiving audio
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    server = await websockets.serve(connection_handler, HOST, PORT)
    transcribe_task = asyncio.create_task(transcribe_audio_service())
    # Start and keep the server running
    await server.wait_closed()
    # Start the transcription process
    await transcribe_task

if __name__ == "__main__":
    try:
        asyncio.run(main())  # Proper event loop handling for Python 3.10+
    except KeyboardInterrupt:
        # stop recording
        audio_source.completed_recording()  
        print("Server shutting down.")
