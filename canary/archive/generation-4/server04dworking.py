# Pivoting away from NVIDIA Streaming Conformer-Hybrid Large (stt_en_fastconformer_hybrid_large_streaming_multi)
# Moving final inference to Speech-augmented Language Model (SALM) transformer nvidia/canary-qwen-2.5b

# CLEANUP IDEAS
# move imports to required models loading class
# rename classes send_message_to_clients() to send_message_to_client()
# rename "NeMo" model"s classes/functions for more explicit
# Write utterances to files in non-blocking manner for testing
import asyncio
from asyncio import Queue
import websockets #pip install websockets
import io
import wave
from pydub import AudioSegment
import datetime
import json
from piper.voice import PiperVoice #pip install piper-tts
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import numpy as np
from pathlib import Path
import uuid
import audioop
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
from nemo.collections.speechlm2.models import SALM
import copy
import time
import atexit  # To handle exits
import struct  # For writing binary data to WAV
import librosa


# Establish the preferred device to run models on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if (DEVICE != "cuda"):
    print('Warning! GPU not detected!')

CONFIG = {
    "websocket_host": "0.0.0.0",
    "websocket_port": 9001,
    "inference_device": DEVICE,
    "piper_model_path": "/root/fawkes/models/piper_tts/en_GB-northern_english_male-medium.onnx",
    "xtts_model_dir": "/root/fawkes/models/coqui_xtts/XTTS-v2/",
    "speakers_dir": "speakers",
    "nemo_model_path": "/root/fawkes/models/fc-hybrid-lg-multi/stt_en_fastconformer_hybrid_large_streaming_multi.nemo",
    "nemo_vad_model_path": "/root/fawkes/models/marblenet_vad_multi/frame_vad_multilingual_marblenet_v2.0.nemo",
    "nemo_encoder_step_length": 80,
    "nemo_lookahead_size": 480, # 0ms, 80ms, 480ms, 1040ms lookahead / 80ms, 160ms, 540ms, 1120ms chunk size
    "nemo_decoder_type": 'rnnt',
    "audio_sample_rate": 16000,
    "vad_sample_rate": 16000,
    "vad_threshold": 0.3, # Increased VAD threshold - COMMON FIX
    "silence_duration_for_finality_ms": 500, # ms of silence to trigger finality
    "canary_qwen_model_path": "/root/fawkes/models/canary-qwen-2.5b/"
}

active_websockets = {}  # client_id -> websocket
client_queues = {}
clientSideTTS = False

# Dialogue partners
SPEAKER = "Nathanael"
SERVER = "Fawkes"

class NemoStreamingTranscriber:
    def __init__(self, model_path, decoder_type, lookahead_size, encoder_step_length, device, sample_rate):
        self.device = device
        self.sample_rate = sample_rate
        self.encoder_step_length = encoder_step_length
        self.model_path = model_path
        self.decoder_type = decoder_type
        self.lookahead_size = lookahead_size
        #self.asr_model = self._load_model()
        # Load the streaming ASR model
        self.asr_model = self._load_streaming_model()
        # Load a separate, dedicated model for offline/n-best transcription
        #self.asr_offline_model = self._load_offline_model()
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

    def _load_streaming_model(self):
        print("Pre-loading NVIDIA NeMo Streaming Conformer-Hybrid Large...")
        asr_model = EncDecHybridRNNTCTCBPEModel.restore_from(self.model_path, map_location=torch.device(self.device))
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
        print("NVIDIA NeMo Streaming Conformer-Hybrid Large loaded successfully")
        return asr_model

    def _load_offline_model(self):
        print("Pre-loading NeMo Offline ASR model for n-best list...")
        asr_model = EncDecHybridRNNTCTCBPEModel.restore_from(self.model_path, map_location=torch.device(self.device))
        asr_model.eval()
        # Set beam search decoding and other parameters once, at load time
        beam_size = 5  # You can adjust this value
        offline_decoding_cfg = asr_model.cfg.decoding
        with open_dict(offline_decoding_cfg):
            offline_decoding_cfg.strategy = "beam"
            offline_decoding_cfg.beam.beam_size = beam_size
            offline_decoding_cfg.beam.return_best_hypothesis = False # This is key for n-best
            # You may want to set preserve_alignments=True if you need word timestamps for the rescorer
            offline_decoding_cfg.preserve_alignments = False
            if hasattr(asr_model, 'joint'):
                offline_decoding_cfg.greedy.max_symbols = 10
                offline_decoding_cfg.fused_batch_size = -1
        asr_model.change_decoding_strategy(offline_decoding_cfg)
        print("NeMo Offline ASR model for n-best list loaded successfully")
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

class CanaryQwenTranscriber:
    def __init__(self, model_path, device):
        print("Pre-loading Canary-Qwen-2.5b model...")
        self.device = device
        self.model_path = Path(model_path)
        
        # Import NeMo's SALM model
        #from nemo.collections.speechlm2.models import SALM
        
        try:
            # Method 1: Try loading from local HuggingFace format (.safetensors + config.json)
            self.model = SALM.from_pretrained(str(self.model_path))
            print("Loaded from HuggingFace format (.safetensors)")
        except Exception as e1:
            print(f"HuggingFace format loading failed: {e1}")
            try:
                # Method 2: Try NeMo restore_from if there's a .nemo file
                nemo_files = list(self.model_path.glob("*.nemo"))
                if nemo_files:
                    self.model = SALM.restore_from(str(nemo_files[0]))
                    print("Loaded from .nemo format")
                else:
                    raise FileNotFoundError("No .nemo file found")
            except Exception as e2:
                print(f"NeMo format loading failed: {e2}")
                try:
                    # Method 3: Manual loading with explicit config
                    config_path = self.model_path / "config.json"
                    model_file = self.model_path / "model.safetensors"
                    
                    if config_path.exists() and model_file.exists():
                        # Load using the config file path directly
                        self.model = SALM.from_pretrained(
                            pretrained_model_name=str(self.model_path),
                            local_files_only=True,
                            trust_remote_code=False
                        )
                        print("Loaded using explicit local config")
                    else:
                        raise FileNotFoundError("Required model files not found")
                except Exception as e3:
                    raise RuntimeError(f"All loading methods failed: {e1}, {e2}, {e3}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store the audio locator tag for convenience
        self.audio_locator_tag = getattr(self.model, 'audio_locator_tag', '<|audio|>')
        
        print("Canary-Qwen-2.5b model loaded successfully")

    def transcribe_final(self, audio_float32, sample_rate=16000):
        """
        Perform final transcription with ITN and P&C using Canary-Qwen-2.5b
        Uses in-memory processing only - no disk I/O
        
        Args:
            audio_float32 (np.ndarray): Audio data as float32 normalized to [-1, 1]
            sample_rate (int): Sample rate of the audio
            
        Returns:
            str: Final transcription with punctuation and capitalization
        """
        try:
            # Ensure audio is in the correct format
            if audio_float32.dtype != np.float32:
                audio_float32 = audio_float32.astype(np.float32)
            
            # Canary-Qwen expects 16kHz audio
            if sample_rate != 16000:
                audio_float32 = librosa.resample(audio_float32, orig_sr=sample_rate, target_sr=16000)
            
            # Method 1: Try direct numpy array transcription
            try:
                # Many NeMo models accept numpy arrays directly
                transcription = self.model.transcribe([audio_float32])
                if isinstance(transcription, list):
                    return transcription[0].strip()
                return transcription.strip()
                        
            except (AttributeError, TypeError, RuntimeError):
                # Method 2: Try with tensor input
                try:
                    audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(self.device)
                    audio_length = torch.tensor([len(audio_float32)]).to(self.device)
                    
                    with torch.no_grad():
                        transcription = self.model.transcribe(
                            audio_signal=audio_tensor,
                            audio_signal_length=audio_length
                        )
                    
                    if isinstance(transcription, list):
                        return transcription[0].strip()
                    return transcription.strip()
                    
                except (AttributeError, TypeError, RuntimeError):
                    # Method 3: Use SALM generate method with in-memory audio
                    try:
                        # Create in-memory audio representation
                        # Some SALM models can work with raw audio data in prompts
                        
                        # Convert audio to the expected format
                        audio_dict = {
                            "audio_data": audio_float32,
                            "sample_rate": 16000
                        }
                        
                        prompts = [
                            [{
                                "role": "user", 
                                "content": f"Transcribe the following: {self.audio_locator_tag}",
                                "audio_signal": audio_tensor,
                                "audio_signal_length": audio_length
                            }]
                        ]
                        
                        with torch.no_grad():
                            answer_ids = self.model.generate(
                                prompts=prompts,
                                max_new_tokens=128,
                                do_sample=False,
                                num_beams=1,  # Greedy decoding for speed
                                temperature=1.0,
                            )
                        
                        # Decode the result
                        transcription = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
                        return transcription.strip()
                        
                    except Exception as inner_e:
                        print(f"All transcription methods failed. Last error: {inner_e}")
                        return ""
                
        except Exception as e:
            print(f"Error in Canary-Qwen transcription: {e}")
            return ""

    def transcribe_with_beam_search(self, audio_float32, sample_rate=16000, num_beams=3):
        """
        Alternative method with beam search for potentially better quality
        """
        try:
            if audio_float32.dtype != np.float32:
                audio_float32 = audio_float32.astype(np.float32)
            
            if sample_rate != 16000:
                audio_float32 = librosa.resample(audio_float32, orig_sr=sample_rate, target_sr=16000)
            
            audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(self.device)
            audio_length = torch.tensor([len(audio_float32)]).to(self.device)
            
            prompts = [
                [{
                    "role": "user", 
                    "content": f"Transcribe the following: {self.audio_locator_tag}",
                    "audio_signal": audio_tensor,
                    "audio_signal_length": audio_length
                }]
            ]
            
            with torch.no_grad():
                answer_ids = self.model.generate(
                    prompts=prompts,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=num_beams,  # Enable beam search
                    temperature=1.0,
                )
            
            transcription = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
            return transcription.strip()
            
        except Exception as e:
            print(f"Error in beam search transcription: {e}")
            return ""

class NeMoVAD:
    def __init__(self, model_path, device, sample_rate=16000):
        print("Pre-loading NeMo VAD model...")
        self.device = device
        self.sample_rate = sample_rate
        # The VAD model is an EncDecClassificationModel
        # Replace with EncDecSpeakerLabelModel older model deprecated
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(model_path, map_location=torch.device(self.device))
        self.model.eval()
        self.model.to(self.device)
        print("NeMo VAD model loaded successfully")

    def detect_voice(self, audio_chunk_int16: np.ndarray):
        # Ensure the audio chunk is 1D
        if audio_chunk_int16.ndim > 1:
            audio_chunk_int16 = audio_chunk_int16.squeeze()

        # Convert to float32 in range [-1.0, 1.0]
        audio_signal = torch.from_numpy(audio_chunk_int16.astype(np.float32) / 32768.0).unsqueeze(0).to(self.device)
        audio_signal_len = torch.Tensor([audio_signal.shape[1]]).to(self.device)

        with torch.no_grad():
            # Get logits from the model
            logits = self.model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
            
            # MarbleNet outputs logits for speech vs non-speech
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=-1)
            
            # Take the speech probability (usually index 1, non-speech is index 0)
            if probabilities.shape[-1] > 1:
                speech_prob = probabilities[..., 1]  # Speech class
            else:
                speech_prob = torch.sigmoid(logits.squeeze())
            
            # Average probability across time frames
            avg_speech_prob = speech_prob.mean().cpu().numpy()
            
            VAD_THRESHOLD = CONFIG["vad_threshold"]  # from config
            is_voice_active = avg_speech_prob > VAD_THRESHOLD
            
            # Debug output (remove after testing)
            #print(f"VAD: avg_prob={avg_speech_prob:.3f}, threshold={VAD_THRESHOLD}, active={is_voice_active}")
            
            return is_voice_active

class PiperTTS:
    def __init__(self, model_path):
        print("Pre-loading Piper TTS model...")
        self.voice = PiperVoice.load(model_path)
        print("Piper TTS model loaded successfully")

    def synthesize_stream_raw(self, text):
        for chunk in self.voice.synthesize_stream_raw(text):
            yield chunk
        yield None # End of stream marker

    @property
    def sample_rate(self):
        return self.voice.config.sample_rate

class XTTSWrapper:
    """
    Encapsulates the Coqui XTTS model, handling model loading, speaker management,
    and raw audio stream inference.
    """
    def __init__(self, model_dir, device, speakers_dir):
        """
        Initializes the XTTS model and speaker manager.

        Args:
            model_dir (Path): Path to the directory containing XTTS model files (config.json, model.pth, vocab.json).
            speakers_dir (Path): Path to the directory containing speaker audio files.
            device (str): Device to load the model on ('cuda' for GPU, 'cpu' for CPU).
        """
        print("Pre-loading Coqui XTTS model...")
        self.device = device
        self.model_dir = Path(model_dir)
        self.config_path = self.model_dir / "config.json"
        self.speakers_dir = Path(speakers_dir)
        self.config = XttsConfig()
        self.config.load_json(self.config_path)
        self.xtts_model = Xtts.init_from_config(self.config)
        self.xtts_model.load_checkpoint(config=self.config, checkpoint_dir=self.model_dir, eval=True)
        self.xtts_model.to(self.device)
        self._load_speakers()

        print(f"Coqui XTTS Model loaded on device: {self.device}")
        #self._load_speakers()
        #print(f"Loaded {len(self.xtts_model.speaker_manager.speakers)} speakers.")

    @property
    def sample_rate(self) -> int:
        """Returns the sample rate of the XTTS model."""
        return self.config.audio["sample_rate"]

    def _load_speakers(self):
        """
        Loads speaker embeddings from the specified speakers_dir into the XTTS model's
        speaker manager. This is called during initialization.
        """
        self.xtts_model.speaker_manager.speakers = {}
        for pt_file in self.speakers_dir.glob("*.pt"):
            data = torch.load(pt_file, map_location="cpu")
            self.xtts_model.speaker_manager.speakers[pt_file.stem] = {
                "gpt_cond_latent": data["gpt_cond_latent"],
                "speaker_embedding": data["speaker_embedding"]
            }
        print(f"Loaded {len(self.xtts_model.speaker_manager.speakers)} speakers.")

    def synthesize_stream_raw(self, text: str, speaker_name: str):
        """
        Performs raw XTTS inference and yields audio chunks.
        This method is synchronous and yields PyTorch tensors or NumPy arrays.
        It does NOT handle client-specific queues or network streaming.

        Args:
            text (str): The text to synthesize.
            speaker_name (str): The name of the speaker to use.

        Yields:
            numpy.ndarray: Raw audio chunks (float32, 24kHz) as NumPy arrays.
                           These will need further processing (e.g., 16-bit PCM conversion,
                           resampling if target is different) by an orchestrator.
        Raises:
            ValueError: If the speaker is not found.
        """
        speaker_data = self.xtts_model.speaker_manager.speakers.get(speaker_name)
        if speaker_data is None:
            raise ValueError(f"Speaker '{speaker_name}' not found.")

        # Ensure speaker embeddings are on the correct device
        gpt_cond_latent = speaker_data["gpt_cond_latent"].to(self.device)
        speaker_embedding = speaker_data["speaker_embedding"].to(self.device)

        # The inference_stream from Coqui TTS typically returns a synchronous generator
        # of torch.Tensor chunks.
        for chunk in self.xtts_model.inference_stream(
            text=text,
            language="en", # Assuming English, adjust if your app supports multiple
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            stream_chunk_size=512, # You can adjust this for latency vs. throughput
        ):
            # Convert PyTorch tensor to NumPy array on CPU before yielding
            # (assuming subsequent processing expects NumPy)
            yield chunk.cpu().numpy()

async def websocket_server(websocket, client_id, nemo_transcriber, nemo_vad, canary_qwen_transcriber):
    active_websockets[client_id] = websocket
    client_queues[client_id] = {
        "incoming_audio": asyncio.Queue(),
        "outgoing_audio": asyncio.Queue(),
        "outgoing_text": asyncio.Queue(),
    }
    try:
        incoming_task = asyncio.create_task(handle_incoming(websocket, client_id))
        outgoing_task = asyncio.create_task(handle_outgoing(websocket, client_id))
        asr_task = asyncio.create_task(process_audio_from_queue(client_id, nemo_transcriber, nemo_vad, canary_qwen_transcriber))
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
    # Handle None or empty chunks upfront for all types
    if chunk is None or (isinstance(chunk, (bytes, np.ndarray)) and len(chunk) == 0):
        return b'' # Always return empty bytes for empty input
    try:
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
                    chunk = librosa.resample(chunk, orig_sr=rate, target_sr=TARGET_RATE)
                # Convert float32 [-1.0, 1.0] to int16 PCM
                chunk = np.clip(chunk, -1.0, 1.0)
                chunk_int16 = (chunk * 32767).astype(np.int16)
                return chunk_int16.tobytes()
            else:
                raise ValueError("Expected NumPy float32 array for type='float32'")
                return b''
        else:
            raise ValueError(f"Unsupported audio type: {type}")
    except Exception as e:
        print(f"Error in prepare_for_streaming for type '{type}': {e}. Returning empty bytes.")
        return b'' # Catch any unexpected errors during processing and return empty bytes

async def send_message_to_clients(client_id, message):
    """Send a message to all connected WebSocket clients."""
    client_queues[client_id]["outgoing_text"].put_nowait(message)

async def stream_tts_audio(client_id, text):
    global pipertts_wrapper, main_loop # Ensure main_loop is globally accessible if not already
    print(f"[TTS] Streaming (raw) to client {client_id}: {text}")

    if client_id not in client_queues:
        print(f"[TTS] Client {client_id} not in client_queues")
        return

    # Define a synchronous function to run in a separate thread
    def blocking_piper_inference():
        try:
            # Get the synchronous generator from Piper TTS
            piper_chunks_generator = pipertts_wrapper.synthesize_stream_raw(text)

            # Iterate over the synchronous generator within this thread
            for chunk in piper_chunks_generator:
                # Prepare the chunk for streaming (this is also blocking if prepare_for_streaming isn't async)
                converted_chunk = prepare_for_streaming(chunk, 'raw', pipertts_wrapper.sample_rate)

                # Only put non-empty chunks into the queue
                if converted_chunk:
                    # Use run_coroutine_threadsafe to put the chunk into the asyncio queue
                    # from this synchronous thread.
                    asyncio.run_coroutine_threadsafe(
                        client_queues[client_id]["outgoing_audio"].put(converted_chunk),
                        main_loop # Reference to the main event loop
                    )

                else:
                    #print(f"Skipping empty converted_chunk from Piper for client {client_id}")
                    pass

            # After all chunks are processed, send the end-of-stream marker
            asyncio.run_coroutine_threadsafe(
                client_queues[client_id]["outgoing_audio"].put(None),
                main_loop
            )
            print(f"[TTS] Finished streaming to client {client_id} (from thread).")

        except Exception as e:
            print(f"[TTS] Error during Piper inference in thread for {client_id}: {e}")
            # Ensure end-of-stream marker is sent even on error
            asyncio.run_coroutine_threadsafe(
                client_queues[client_id]["outgoing_audio"].put(None),
                main_loop
            )

    try:
        # Offload the entire blocking synchronous inference process to a separate thread
        await asyncio.to_thread(blocking_piper_inference)

    except Exception as e:
        # This outer catch will only catch errors related to setting up the thread,
        print(f"[TTS] Error setting up Piper streaming to {client_id}: {e}")

async def synthesize_xtts_audio(speaker_name, text, buffer):
    """
    Kicks off XTTS inference in a background thread and puts processed audio chunks
    into the provided asyncio.Queue buffer.
    """
    global xtts_wrapper, main_loop
    print(f"Computing XTTS for: {text}")

    def blocking_direct_inference():
        try:
            # Call the XTTSWrapper's raw generator directly
            chunks_raw = xtts_wrapper.synthesize_stream_raw(text, speaker_name)

            for chunk_np_float32 in chunks_raw: # chunk_np_float32 is already a NumPy array from synthesize_stream_raw
                if chunk_np_float32 is None or len(chunk_np_float32) == 0:
                    continue

                # Prepare the chunk for streaming (e.g., to 16-bit PCM bytes)
                # Use xtts_wrapper.sample_rate for consistency
                converted_chunk = prepare_for_streaming(chunk_np_float32, 'float32', xtts_wrapper.sample_rate)

                # Send it to the buffer queue from this thread (thread-safe way)
                asyncio.run_coroutine_threadsafe(
                    buffer.put(converted_chunk),
                    main_loop
                )
            # Signal end-of-stream
            asyncio.run_coroutine_threadsafe(
                buffer.put(None),
                main_loop
            )
            print(f"[XTTS] Finished streaming to buffer.") # client_id not directly in this function now
        except ValueError as ve:
            print(f"[XTTS] Speaker Error during inference: {ve}")
            asyncio.run_coroutine_threadsafe(buffer.put(None), main_loop)
        except Exception as e:
            print(f"[XTTS] General Error during inference: {e}")
            asyncio.run_coroutine_threadsafe(buffer.put(None), main_loop)

    # Run the blocking work in a background thread
    await asyncio.to_thread(blocking_direct_inference)

async def stream_xtts_audio(client_id, buffer):
    """
    Streams pre-buffered XTTS audio chunks from an asyncio.Queue to a specific client's
    outgoing_audio queue.
    """
    print(f"[XTTS] Streaming from buffer queue to client {client_id}")
    queue = client_queues.get(client_id, {}).get("outgoing_audio")
    if not queue:
        print(f"[XTTS] Client {client_id} not in client_queues or missing 'outgoing_audio'")
        return
    try:
        while True:
            chunk = await buffer.get()
            if chunk is None:  # End-of-stream marker
                break
            await queue.put(chunk)
            await asyncio.sleep(0.015)
        await queue.put(None)
    except Exception as e:
        print(f"[XTTS] Error streaming to {client_id}: {e}")

async def synthesize_stream_xtts_audio(client_id, speaker_name, text):
    """
    Generates speech using Coqui XTTS and streams it directly over WebSockets
    to a specific client without pre-buffering.
    """
    global xtts_wrapper, main_loop
    print(f"Streaming XTTS directly for: {text}")

    # This is the client's outgoing audio queue
    client_queue = client_queues[client_id]["outgoing_audio"]

    def blocking_direct_inference():
        """
        This synchronous function runs in a separate thread,
        iterates the XTTS generator, and puts chunks into the asyncio queue.
        """
        try:
            # Get the synchronous generator from the XTTSWrapper
            chunks_raw_generator = xtts_wrapper.synthesize_stream_raw(text, speaker_name)

            # Iterate over the synchronous generator in this background thread
            for chunk_np_float32 in chunks_raw_generator:
                if chunk_np_float32 is None or len(chunk_np_float32) == 0:
                    continue # Skip empty/None chunks

                # Prepare the chunk for streaming in this background thread
                converted_chunk = prepare_for_streaming(chunk_np_float32, 'float32', xtts_wrapper.sample_rate)

                # Send the processed chunk to the client's queue from this thread.
                # asyncio.run_coroutine_threadsafe is used for thread-safe scheduling
                # of a coroutine (queue.put) onto the main event loop.
                asyncio.run_coroutine_threadsafe(
                    client_queue.put(converted_chunk),
                    main_loop
                )

            # After all chunks are processed, signal end-of-stream to the client's queue
            asyncio.run_coroutine_threadsafe(
                client_queue.put(None),
                main_loop
            )
            print(f"[XTTS] Finished direct streaming to client {client_id} (from thread).")

        except ValueError as ve:
            # Catch specific ValueErrors (like speaker not found from XTTSWrapper)
            print(f"[XTTS] Speaker/Input Error during direct streaming to {client_id} (in thread): {ve}")
            asyncio.run_coroutine_threadsafe(client_queue.put(None), main_loop) # Signal end on error
        except Exception as e:
            # Catch any other unexpected errors
            print(f"[XTTS] General Error during direct streaming to {client_id} (in thread): {e}")
            asyncio.run_coroutine_threadsafe(client_queue.put(None), main_loop) # Signal end on error

    # Run the entire blocking_direct_inference function in a separate thread
    await asyncio.to_thread(blocking_direct_inference)

async def process_audio_from_queue(client_id, nemo_transcriber, nemo_vad, canary_qwen_transcriber):
    """
    Processes audio chunks from an asyncio.Queue.
    """
    chunk_size_ms = CONFIG["nemo_lookahead_size"] + CONFIG["nemo_encoder_step_length"]
    bytes_per_chunk = int(CONFIG["audio_sample_rate"] * chunk_size_ms / 1000) * 2  # 2 bytes per sample (int16)
    audio_buffer = b''  # Initialize an empty byte buffer

    # New VAD-related state variables
    current_utterance_buffer = b''
    is_speaking = False
    silence_counter = 0
    # Determine silence threshold in terms of VAD chunks.
    # If your VAD processes in, say, 20ms frames, and your audio_chunk is 560ms,
    # then one audio_chunk corresponds to 28 VAD frames.
    # A VAD threshold of ~0.5 to 1 second of continuous silence is usually good for finality.
    # So, 1 second / (chunk_size_ms / 1000) = chunks per second
    # For 560ms chunks, it's ~1.78 chunks/sec. 3 chunks ~ 1.68s. Adjust as needed.
    SILENCE_CHUNKS_THRESHOLD = 2 # Number of consecutive silent ASR chunks to consider end of utterance
    #previous_text = ""  # Store the previously sent text (for incremental updates)
    final_transcription_text = ""

    try:
        while True:
            try:
                audio_data = await client_queues[client_id]["incoming_audio"].get()
                if audio_data is None or len(audio_data) == 0:
                    await asyncio.sleep(0.001)  # Or handle empty chunk as appropriate
                    continue  # Skip processing empty chunk
                audio_buffer += audio_data
                while len(audio_buffer) >= bytes_per_chunk:
                    chunk_bytes = audio_buffer[:bytes_per_chunk]
                    audio_buffer = audio_buffer[bytes_per_chunk:]
                    audio_chunk_np = np.frombuffer(chunk_bytes, dtype=np.int16)
                    # Ensure the audio_chunk_np is 1D for VAD
                    if audio_chunk_np.ndim != 1:
                        audio_chunk_np = audio_chunk_np.squeeze()
                    #if audio_chunk_np.dtype != np.int16:
                    #    audio_chunk_np = audio_chunk_np.astype(np.int16)
                    
                    # --- VAD Detection ---
                    # Run VAD inference in a thread pool to avoid blocking the event loop
                    is_voice_active_in_chunk = await asyncio.to_thread(nemo_vad.detect_voice, audio_chunk_np)

                    #print(f"Audio chunk stats: min={audio_chunk_np.min()}, max={audio_chunk_np.max()}, std={audio_chunk_np.std()}")

                    if is_voice_active_in_chunk:
                        current_utterance_buffer += chunk_bytes # Accumulate all speech
                        silence_counter = 0 # Reset silence counter

                        if not is_speaking:
                            is_speaking = True
                            #print(f"[{client_id}] Voice activity started.")
                            # Reset ASR state for a new utterance
                            nemo_transcriber.previous_hypotheses = None
                            nemo_transcriber.pred_out_stream = None
                            nemo_transcriber.step_num = 0
                            num_channels = nemo_transcriber.asr_model.cfg.preprocessor.features
                            nemo_transcriber.cache_pre_encode = torch.zeros((1, num_channels, nemo_transcriber.pre_encode_cache_size),
                                                                        device=nemo_transcriber.device)
                            nemo_transcriber.cache_last_channel, nemo_transcriber.cache_last_time, nemo_transcriber.cache_last_channel_len = \
                                nemo_transcriber.asr_model.encoder.get_initial_cache_state(batch_size=1)
                            text = ""
                            final_transcription_text = ""  # Reset for new utterance

                        # Perform ASR transcription on the *current audio chunk* if speech is active
                        text = await asyncio.to_thread(nemo_transcriber.transcribe_chunk, audio_chunk_np)
                        final_transcription_text = text  # Keep updating the final transcription

                        data_to_send = {
                            "speaker": SPEAKER,
                            "final": False, # Always interim while speaking
                            "transcript": text
                        }
                        json_string = json.dumps(data_to_send)
                        await send_message_to_clients(client_id, json_string)

                    else: # VAD indicates silence
                        silence_counter += 1
                        if is_speaking and silence_counter >= SILENCE_CHUNKS_THRESHOLD:
                            #is_speaking = False
                            #print(f"[{client_id}] Voice activity ended. Processing final utterance.")
                            print("Acoustic finality detected. Processing full utterance with offline model...")
                            '''
                            # We now have accoustic finality. Perform final offline n-best beam search
                            # Then send to rescorer and P&C to determine linguistic finality

                            # Get the bytes from the current_utterance_buffer
                            audio_bytes = current_utterance_buffer
                            # Convert the raw bytes to a numpy array of 16-bit integers
                            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                            # Convert the 16-bit integers to 32-bit floats and normalize (as is standard for ASR)
                            audio_float32 = audio_int16.astype(np.float32) / 32768.0

                            # Use Canary-Qwen for final transcription with ITN and P&C
                            final_canary_transcription = await asyncio.to_thread(
                                canary_qwen_transcriber.transcribe_final, 
                                audio_float32,
                                CONFIG["audio_sample_rate"]
                            )
                            print(f"Canary-Qwen final transcription: {final_canary_transcription}")

                            # Use Canary-Qwen result as the final transcription
                            if final_canary_transcription:
                                final_transcription_text = final_canary_transcription
                            '''
                            # OLD METHOD Use the last transcription result
                            if final_transcription_text:
                                data_to_send = {
                                    "speaker": SPEAKER,
                                    "final": True,
                                    "transcript": final_transcription_text
                                }
                                json_string = json.dumps(data_to_send)
                                await send_message_to_clients(client_id, json_string)

                                # Trigger Time/Name responses only on final utterances from recognized SPEAKER
                                if 'the time' in final_transcription_text.lower():
                                    print("Asked about the time")
                                    strTime = datetime.datetime.now().strftime("%H:%M:%S")
                                    response_text = f"Sir, the time is {strTime}"
                                    data_to_send = {
                                        "speaker": SERVER,
                                        "final": "True",
                                        "transcript": response_text
                                    }
                                    json_string = json.dumps(data_to_send)
                                    await send_message_to_clients(client_id, json_string)
                                    if not clientSideTTS and active_websockets:
                                        asyncio.create_task(stream_tts_audio(client_id, response_text))
                                elif 'your name' in final_transcription_text.lower():
                                    print("Asked about my name")
                                    response_text = "My name is Neil Richard Gaiman."
                                    data_to_send = {
                                        "speaker": SERVER,
                                        "final": "True",
                                        "transcript": response_text
                                    }
                                    json_string = json.dumps(data_to_send)
                                    if active_websockets:
                                        await send_message_to_clients(client_id, json_string)
                                    if not clientSideTTS and active_websockets:
                                        async def parallel_tts_pipeline():
                                            buffer = asyncio.Queue()
                                            coqui_task = asyncio.create_task(synthesize_xtts_audio("neil_gaiman", response_text, buffer))
                                            await stream_tts_audio(client_id, "Compiling response, please wait a moment...")
                                            await stream_xtts_audio(client_id, buffer)
                                            await coqui_task
                                        asyncio.create_task(parallel_tts_pipeline())
                                
                            # Reset the buffer and state for the next utterance
                            is_speaking = False
                            silence_counter = 0
                            current_utterance_buffer = b''
                            #previous_text = "" # Reset previous_text for the new utterance
                            text = ""

            except asyncio.QueueEmpty:  # asyncio uses asyncio.QueueEmpty
                await asyncio.sleep(0.01)  # Use asyncio.sleep
            except Exception as e:
                print(f"Error processing audio for {client_id}: {e}")
                break
            finally:
                #client_queues[client_id]["incoming_audio"].task_done() # Necessary for asyncio.Queue
                pass

    finally:
        print("Async Audio processing stopped")

async def connection_handler(websocket):
    global client_id, nemo_transcriber, nemo_vad, canary_qwen_transcriber
    # For multiclient support (direct instantiation) you would actually initialize stateful models per-client here (EXPENSIVE)
    # Realistically would need several "free-floating" instances that could be quick-attached to new streams (esp. w/ speech separation)
    # Long-term: add model serving framework (e.g. NVIDIA Triton Inference Server, TorchServe, FastAPI/MLflow with custom logic)
    # Shared weights, separate contexts - maintain separate internal states for each client
    # Dynamic batching with clients' individual states managed separately by the serving layer
    # Triton can manage transformer and SSM models, can also help organize adapter layers
    client_id = str(uuid.uuid4())
    print(f"New client connected: {client_id}")
    await websocket_server(websocket, client_id, nemo_transcriber, nemo_vad, canary_qwen_transcriber)

async def main():
    global main_loop, nemo_transcriber, pipertts_wrapper, xtts_wrapper, nemo_vad, canary_qwen_transcriber
    main_loop = asyncio.get_event_loop()  # Store the event loop
    # Initialize PiperTTS instance
    pipertts_wrapper = PiperTTS(CONFIG["piper_model_path"])
    # Initialize Coqui XTTS instance
    #xtts_wrapper = XTTSWrapper(CONFIG["xtts_model_dir"], CONFIG["inference_device"], CONFIG["speakers_dir"])
    xtts_wrapper = None
    # Initialize the NVIDIA Streaming Conformer-Hybrid Large transcriber (for interim results)
    nemo_transcriber = NemoStreamingTranscriber(
        model_path=CONFIG["nemo_model_path"],
        decoder_type=CONFIG["nemo_decoder_type"],
        lookahead_size=CONFIG["nemo_lookahead_size"],
        encoder_step_length=CONFIG["nemo_encoder_step_length"],
        device=CONFIG["inference_device"],
        sample_rate=CONFIG["audio_sample_rate"]
    )
    # Initialize Canary-Qwen-2.5b transcriber (for final results)
    '''
    canary_qwen_transcriber = CanaryQwenTranscriber(
        model_path=CONFIG["canary_qwen_model_path"],
        device=CONFIG["inference_device"]
    )
    '''
    canary_qwen_transcriber = None
    # Initialize the NeMo VAD model
    nemo_vad = NeMoVAD(
        model_path=CONFIG["nemo_vad_model_path"],
        device=CONFIG["inference_device"],
        sample_rate=CONFIG["vad_sample_rate"]
    )
    # Start the WebSocket server for receiving audio
    print(f"Starting WebSocket server on ws://{CONFIG['websocket_host']}:{CONFIG['websocket_port']}")
    server = await websockets.serve(connection_handler, CONFIG['websocket_host'], CONFIG['websocket_port'])
    # Start and keep the server running
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())  # Proper event loop handling for Python 3.10+
    except KeyboardInterrupt: 
        print("Server shutting down.")