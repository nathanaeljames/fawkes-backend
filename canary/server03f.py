# Fawkes Voice AI Server v03 - Refactored monolith
# Changes from server02f.py:
#   - ServerContext replaces all global state (no more `global` statements)
#   - DatabaseManager class owns DB lifecycle
#   - AudioUtils class for stateless audio conversions
#   - MessageRouter class for client message construction/delivery
#   - TTSStreamManager class absorbs all TTS streaming functions
#   - WebSocketManager class absorbs websocket lifecycle and audio processing loop
#   - RasaHandler extends RasaClient with response processing
#   - All classes receive ctx reference instead of reaching for globals
#   - Removed dead code: _load_model(), duplicate imports, inline import json
#   - Removed commented-out testing overrides
#   - Kept save_utterance_async and manual_sequential_ecapa as utility functions
#
# NOTES
# May circle back for Lexical/Audio P&C using NeMo's models (Canary is underperforming)
# May also experiment queueing multiple utterances in a floating frame to increase context available to Canary-Qwen/Samba
# Samba-ASR adoption may require language model fusion/contextual rescoring/full LLM integration/P&C

from __future__ import annotations

import asyncio
from asyncio import Queue
import datetime
import logging
import websockets
import io
import wave
from pydub import AudioSegment
import json
from piper.voice import PiperVoice
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import numpy as np
from pathlib import Path
import uuid
import audioop
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.speechlm2.models import SALM
import copy
import time
import atexit
import librosa
from scipy.io.wavfile import write as wav_write
import duckdb
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional, List, Dict, Any, BinaryIO, Union
import torchaudio
import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import uvicorn
import random
from difflib import get_close_matches, SequenceMatcher
import re
import traceback


# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    logging.warning('GPU not detected!')

CONFIG = {
    # --- Server ---
    "websocket_host": "0.0.0.0",
    "websocket_port": 9001,
    "fastapi_host": "0.0.0.0",
    "fastapi_port": 9002,

    # --- Model paths ---
    "inference_device": DEVICE,
    "piper_model_path": "/root/fawkes/models/piper_tts/en_GB-northern_english_male-medium.onnx",
    "xtts_model_dir": "/root/fawkes/models/coqui_xtts/XTTS-v2/",
    "nemo_model_path": "/root/fawkes/models/fc-hybrid-lg-multi/stt_en_fastconformer_hybrid_large_streaming_multi.nemo",
    "nemo_vad_model_path": "/root/fawkes/models/marblenet_vad_multi/frame_vad_multilingual_marblenet_v2.0.nemo",
    "canary_qwen_model_path": "/root/fawkes/models/canary-qwen-2.5b/",
    "ecapa_tdnn_model_path": "/root/fawkes/models/ecapa_tdnn_embed/ecapa_tdnn.nemo",
    "duckdb_path": "./speakers/database.duckdb",

    # --- ASR settings ---
    "nemo_encoder_step_length": 80,
    "nemo_lookahead_size": 480,
    "nemo_decoder_type": 'rnnt',
    "canary_max_new_tokens": 128,          # max output tokens for Canary-Qwen transcription

    # --- Audio ---
    "audio_sample_rate": 16000,
    "vad_sample_rate": 16000,
    "vad_threshold": 0.3,
    "silence_duration_for_finality_ms": 500,
    "silence_chunks_threshold": 2,         # consecutive silent chunks before finality

    # --- ECAPA speaker recognition ---
    "ecapa_uncertain_threshold": 0.70,     # below this, speaker is "unknown"
    "ecapa_certain_threshold": 0.85,       # above this, speaker is confidently identified
    "ecapa_nomatch_lower_threshold": 0.70, # below upper but above lower = "unregistered(?)"
    "ecapa_nomatch_upper_threshold": 0.85, # above this = confidently "unregistered"
    "ecapa_max_extractions": 7,            # max ECAPA extractions per utterance

    # --- TTS ---
    "xtts_language": "en",                 # XTTS inference language
    "xtts_stream_chunk_size": 512,         # XTTS streaming chunk size

    # --- Rasa ---
    "rasa_url": "http://rasa-nlp:5005",
    "rasa_timeout": 10,
    "enable_rasa": True,

    # --- Enrollment ---
    "enrollment_off_topic_threshold": 0.5,
    "enrollment_reminder_interval": 3,
    "enrollment_timeout": 18,
    "enrollment_max_decreases": 3,
    "enrollment_fuzzy_word_threshold": 0.85,
    "enrollment_success_threshold": 0.70,
    "enrollment_exact_match_threshold": 0.90,  # minimum score for exact speaker name match

    # --- Voice cloning ---
    "voice_clone_playback_timeout": 15,    # seconds to wait for client playback confirmation
    "passages_substring_boost": 0.90,      # minimum score when fuzzy source is a substring match

    # --- Misc ---
    "server_name": "Fawkes",
    "default_speaker": "unknown speaker",
    "default_speaker_confidence": "uncertain",
    "default_asr_confidence": "certain",
    "samples_path": "/root/fawkes/audio_samples",
    "speakers_dir": "speakers",
    "log_file": None,   # explicit path, or None to auto-generate logs/fawkes_YYYYMMDD_HHMMSS.log
}


# =============================================================================
# LOGGING
# =============================================================================

_log_path = CONFIG["log_file"] or f"logs/fawkes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
Path(_log_path).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_path),
    ]
)
logger = logging.getLogger("fawkes")


# =============================================================================
# SERVER CONTEXT - Central state container, replaces all global variables
# =============================================================================

class ServerContext:
    """
    Holds all shared server state. Passed to every class that needs access
    to models, client state, database, or configuration.
    Replaces all module-level global variables and `global` statements.
    """
    def __init__(self, config: dict):
        self.config = config
        self.main_loop: asyncio.AbstractEventLoop = None
        self.fastapi_app: FastAPI = None

        # Client state
        self.active_websockets: Dict[str, Any] = {}
        self.client_queues: Dict[str, dict] = {}
        self.audio_playback_complete: Dict[str, bool] = {}
        self.client_side_tts: bool = False

        # Database
        self.db: Optional[DatabaseManager] = None

        # Model references (populated during startup)
        self.piper_tts: Optional[PiperTTS] = None
        self.xtts: Optional[XTTSWrapper] = None
        self.nemo_transcriber: Optional[NemoStreamingTranscriber] = None
        self.nemo_vad: Optional[NemoVAD] = None
        self.canary_qwen: Optional[CanaryQwenTranscriber] = None
        self.ecapa_matcher: Optional[FastECAPASpeakerMatcher] = None
        self.ecapa_processor: Optional[ECAPASpeakerProcessor] = None

        # Handler references (populated during startup)
        self.rasa_handler: Optional[RasaHandler] = None
        self.enrollment_manager: Optional[EnrollmentRecordingManager] = None
        self.enrollment_api: Optional[EnrollmentAPIHandler] = None
        self.voiceclone_api: Optional[VoiceCloneAPIHandler] = None
        self.tts_manager: Optional[TTSStreamManager] = None
        self.ws_manager: Optional[WebSocketManager] = None
        self.msg: Optional[MessageRouter] = None


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """Owns the DuckDB connection lifecycle and schema setup."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.con = duckdb.connect(db_path)
        atexit.register(self.close)

    def setup_tables(self):
        """Create all required tables and sequences."""
        self.con.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_uid START 1;
            CREATE TABLE IF NOT EXISTS speakers (
                uid INTEGER PRIMARY KEY DEFAULT nextval('seq_uid'),
                firstname VARCHAR NOT NULL,
                surname VARCHAR,
                gpt_cond_latent FLOAT[],
                gpt_shape VARCHAR,
                xtts_embedding FLOAT[],
                xtts_shape VARCHAR,
                ecapa_embedding FLOAT[],
                total_duration_sec FLOAT DEFAULT 0.0,
                sample_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                pangrams INTEGER[] DEFAULT []
            );
        """)
        logger.info("DuckDB table 'speakers' is ready.")

        self.con.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_pangram_id START 1;
            CREATE TABLE IF NOT EXISTS pangrams (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_pangram_id'),
                text VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logger.info("DuckDB table 'pangrams' is ready.")

        self.con.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_passage_id START 1;
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_passage_id'),
                source VARCHAR NOT NULL,
                quote TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logger.info("DuckDB table 'passages' is ready.")

    def execute(self, *args, **kwargs):
        """Proxy to underlying connection.execute()."""
        return self.con.execute(*args, **kwargs)

    def close(self):
        """Close the database connection."""
        if self.con:
            self.con.close()
            self.con = None


# =============================================================================
# AUDIO UTILITIES - Stateless audio conversion functions
# =============================================================================

class AudioUtils:
    """Pure stateless audio conversion utilities."""

    TARGET_RATE = 16000
    TARGET_WIDTH = 2       # 16-bit
    TARGET_CHANNELS = 1    # mono

    @staticmethod
    def int16_to_float32(audio_int16: np.ndarray) -> np.ndarray:
        """Convert int16 PCM to float32 in [-1.0, 1.0] range."""
        return audio_int16.astype(np.float32) / 32768.0

    @staticmethod
    def float32_to_int16(audio_float32: np.ndarray) -> np.ndarray:
        """Convert float32 [-1.0, 1.0] to int16 PCM."""
        clipped = np.clip(audio_float32, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)

    @staticmethod
    def prepare_for_streaming(chunk, audio_type, rate):
        """Convert a chunk of audio into 16kHz 16-bit mono PCM for streaming."""
        if chunk is None or (isinstance(chunk, (bytes, np.ndarray)) and len(chunk) == 0):
            return b''
        try:
            if audio_type == "wav":
                audio = AudioSegment.from_file(io.BytesIO(chunk), format="wav")
                audio = audio.set_frame_rate(AudioUtils.TARGET_RATE).set_channels(AudioUtils.TARGET_CHANNELS).set_sample_width(AudioUtils.TARGET_WIDTH)
                return audio.raw_data
            elif audio_type == "raw":
                if rate != AudioUtils.TARGET_RATE:
                    chunk = audioop.ratecv(chunk, AudioUtils.TARGET_WIDTH, AudioUtils.TARGET_CHANNELS, rate, AudioUtils.TARGET_RATE, None)[0]
                return chunk
            elif audio_type == "float32":
                if isinstance(chunk, np.ndarray) and chunk.dtype == np.float32:
                    if rate != AudioUtils.TARGET_RATE:
                        chunk = librosa.resample(chunk, orig_sr=rate, target_sr=AudioUtils.TARGET_RATE)
                    chunk_int16 = AudioUtils.float32_to_int16(chunk)
                    return chunk_int16.tobytes()
                else:
                    raise ValueError("Expected NumPy float32 array for type='float32'")
            else:
                raise ValueError(f"Unsupported audio type: {audio_type}")
        except Exception as e:
            logger.error(f"Error in prepare_for_streaming for type '{audio_type}': {e}. Returning empty bytes.")
            return b''


# =============================================================================
# MESSAGE ROUTER - Client message construction and delivery
# =============================================================================

class MessageRouter:
    """Handles building and sending messages to WebSocket clients."""

    def __init__(self, ctx: ServerContext):
        self.ctx = ctx

    def build_transcript_message(self, speaker, speaker_confidence, final, transcript, asr_confidence="certain"):
        """Build a standard transcript message dict and return as JSON string."""
        data = {
            "speaker": speaker,
            "speaker_confidence": speaker_confidence,
            "final": final,
            "transcript": transcript,
            "asr_confidence": asr_confidence
        }
        return json.dumps(data)

    async def send_to_client(self, client_id, message):
        """Send a text message to a specific client's outgoing queue."""
        if client_id in self.ctx.client_queues:
            self.ctx.client_queues[client_id]["outgoing_text"].put_nowait(message)

    async def send_transcript(self, client_id, speaker, speaker_confidence, final, transcript, asr_confidence="certain"):
        """Build and send a transcript message in one call."""
        msg = self.build_transcript_message(speaker, speaker_confidence, final, transcript, asr_confidence)
        await self.send_to_client(client_id, msg)


# =============================================================================
# ASR MODELS
# =============================================================================

class NemoStreamingTranscriber:
    def __init__(self, model_path, decoder_type, lookahead_size, encoder_step_length, device, sample_rate):
        self.device = device
        self.sample_rate = sample_rate
        self.encoder_step_length = encoder_step_length
        self.model_path = model_path
        self.decoder_type = decoder_type
        self.lookahead_size = lookahead_size
        self.asr_model = self._load_streaming_model()
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

    def _load_streaming_model(self):
        logger.info("Pre-loading NVIDIA NeMo Streaming Conformer-Hybrid Large...")
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
        logger.info("NVIDIA NeMo Streaming Conformer-Hybrid Large loaded successfully")
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
        audio_data = AudioUtils.int16_to_float32(new_chunk)
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
    def __init__(self, model_path, device, max_new_tokens=128):
        logger.info("Pre-loading Canary-Qwen-2.5b model...")
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_path = Path(model_path)
        self.model = SALM.from_pretrained(str(self.model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Canary-Qwen-2.5b model loaded successfully")

    def transcribe_final(self, audio_int16, sample_rate=16000):
        """Perform final transcription with ITN and P&C using Canary-Qwen-2.5b."""
        try:
            logger.debug(f"[Canary-Qwen] Processing audio: shape={audio_int16.shape}, sample_rate={sample_rate}")
            audio_float32 = AudioUtils.int16_to_float32(audio_int16)
            audio_tensor = torch.from_numpy(audio_float32).to(self.device)
            audios = audio_tensor.unsqueeze(0)
            audio_lens = torch.tensor([audios.shape[1]], dtype=torch.int64).to(self.device)
            prompts = [
                [{"role": "user", "content": f"Transcribe the following: {self.model.audio_locator_tag}"}]
            ]
            logger.debug("[Canary-Qwen] Running model.generate() with tensor input...")
            with torch.no_grad():
                raw_output_ids = self.model.generate(
                    prompts=prompts,
                    audios=audios,
                    audio_lens=audio_lens,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0
                )
            raw_result = self.model.tokenizer.ids_to_text(raw_output_ids[0].cpu())
            logger.debug(f"[Canary-Qwen] Raw model output: '{raw_result}'")
            result = raw_result.strip()
            if '<|im_start|>' in result:
                parts = result.split('<|im_start|>assistant\n')
                if len(parts) > 1:
                    result = parts[1].split('<|im_end|>')[0].strip()
            result = result.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
            if not result or result.lower() in ['transcript', 'transcription', 'audio transcript']:
                logger.warning(f"[Canary-Qwen] Got generic/empty response: '{result}'")
                return ""
            logger.info(f"[Canary-Qwen] Final transcription: '{result}'")
            return result
        except Exception as e:
            logger.error(f"[Canary-Qwen] Error in transcription: {e}")
            return ""

    def transcribe_with_beam_search(self, audio_int16, sample_rate=16000, num_beams=3):
        """Alternative method with beam search for potentially better quality."""
        try:
            logger.debug(f"[Canary-Qwen] Running beam search transcription with {num_beams} beams")
            audio_float32 = AudioUtils.int16_to_float32(audio_int16)
            audio_tensor = torch.from_numpy(audio_float32).to(self.device)
            audios = audio_tensor.unsqueeze(0)
            audio_lens = torch.tensor([audios.shape[1]], dtype=torch.int64).to(self.device)
            prompts = [
                [{"role": "user", "content": f"Transcribe the following: {self.model.audio_locator_tag}"}]
            ]
            with torch.no_grad():
                raw_output_ids = self.model.generate(
                    prompts=prompts,
                    audios=audios,
                    audio_lens=audio_lens,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=num_beams,
                    temperature=1.0
                )
            raw_result = self.model.tokenizer.ids_to_text(raw_output_ids[0].cpu())
            logger.debug(f"[Canary-Qwen] Raw beam search output: '{raw_result}'")
            result = raw_result.strip()
            if '<|im_start|>' in result:
                parts = result.split('<|im_start|>assistant\n')
                if len(parts) > 1:
                    result = parts[1].split('<|im_end|>')[0].strip()
            result = result.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
            if not result or result.lower() in ['transcript', 'transcription', 'audio transcript']:
                logger.warning(f"[Canary-Qwen] Beam search got generic/empty response: '{result}'")
                return ""
            logger.info(f"[Canary-Qwen] Beam search transcription: '{result}'")
            return result
        except Exception as e:
            logger.error(f"[Canary-Qwen] Error in beam search transcription: {e}")
            return ""


# =============================================================================
# VAD MODEL
# =============================================================================

class NemoVAD:
    def __init__(self, model_path, device, sample_rate=16000):
        logger.info("Pre-loading NeMo VAD model...")
        self.device = device
        self.sample_rate = sample_rate
        self.model = nemo_asr.models.EncDecClassificationModel.restore_from(
            model_path, map_location=torch.device(self.device), strict=False)
        self.model.eval()
        self.model.to(self.device)
        logger.info("NeMo VAD model loaded successfully")

    def detect_voice(self, audio_chunk_int16: np.ndarray):
        if audio_chunk_int16.ndim > 1:
            audio_chunk_int16 = audio_chunk_int16.squeeze()
        audio_signal = torch.from_numpy(AudioUtils.int16_to_float32(audio_chunk_int16)).unsqueeze(0).to(self.device)
        audio_signal_len = torch.Tensor([audio_signal.shape[1]]).to(self.device)
        with torch.no_grad():
            logits = self.model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
            probabilities = torch.softmax(logits, dim=-1)
            if probabilities.shape[-1] > 1:
                speech_prob = probabilities[..., 1]
            else:
                speech_prob = torch.sigmoid(logits.squeeze())
            avg_speech_prob = speech_prob.mean().cpu().numpy()
            VAD_THRESHOLD = CONFIG["vad_threshold"]
            is_voice_active = avg_speech_prob > VAD_THRESHOLD
            return is_voice_active


# =============================================================================
# TTS MODELS
# =============================================================================

class PiperTTS:
    def __init__(self, model_path):
        logger.info("Pre-loading Piper TTS model...")
        self.voice = PiperVoice.load(model_path)
        logger.info("Piper TTS model loaded successfully")

    def synthesize_stream_raw(self, text):
        for chunk in self.voice.synthesize_stream_raw(text):
            yield chunk
        yield None

    @property
    def sample_rate(self):
        return self.voice.config.sample_rate


class XTTSWrapper:
    """Encapsulates the Coqui XTTS model, handling model loading, speaker management,
    and raw audio stream inference."""

    def __init__(self, ctx: ServerContext, model_dir, device, speakers_dir):
        logger.info("Pre-loading Coqui XTTS model...")
        self.ctx = ctx
        self.device = device
        self.model_dir = Path(model_dir)
        self.config_path = self.model_dir / "config.json"
        self.speakers_dir = Path(speakers_dir)
        self.config = XttsConfig()
        self.config.load_json(self.config_path)
        self.xtts_model = Xtts.init_from_config(self.config)
        self.xtts_model.load_checkpoint(config=self.config, checkpoint_dir=self.model_dir, eval=True)
        self.xtts_model.to(self.device)
        self._load_speakers_from_db()
        logger.info(f"Coqui XTTS Model loaded on device: {self.device}")

    @property
    def sample_rate(self) -> int:
        return self.config.audio["sample_rate"]

    async def extract_xtts_embed(self, wav_path, firstname, surname=None):
        """Extracts XTTS embeddings from a WAV file and stores them in the DuckDB database."""
        wav_path = Path(wav_path)
        logger.info(f"Extracting XTTS embeddings for {firstname} {surname if surname else ''} from {wav_path}...")
        with torch.no_grad():
            gpt_cond_latent, speaker_embedding = self.xtts_model.get_conditioning_latents(str(wav_path), 16000)
        logger.debug(f"Extracted shapes - GPT: {gpt_cond_latent.shape}, Speaker: {speaker_embedding.shape}")
        gpt_latent_flat = gpt_cond_latent.cpu().numpy().flatten().tolist()
        xtts_embedding_flat = speaker_embedding.cpu().numpy().flatten().tolist()
        gpt_shape_json = json.dumps(list(gpt_cond_latent.shape))
        xtts_shape_json = json.dumps(list(speaker_embedding.shape))
        db = self.ctx.db

        def insert_data():
            db.execute("""
                INSERT INTO speakers 
                (firstname, surname, gpt_cond_latent, gpt_shape, xtts_embedding, xtts_shape) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (firstname, surname, gpt_latent_flat, gpt_shape_json, xtts_embedding_flat, xtts_shape_json))

        await asyncio.to_thread(insert_data)
        logger.info(f"Successfully stored XTTS embeddings for {firstname} in DuckDB using native arrays.")

    def _load_speakers_from_db(self):
        """Loads XTTS speaker embeddings from the DuckDB database."""
        logger.info("Loading XTTS speakers from DuckDB native arrays...")
        db = self.ctx.db
        speakers_query = db.execute("""
            SELECT firstname, surname, gpt_cond_latent, gpt_shape, xtts_embedding, xtts_shape 
            FROM speakers 
            WHERE gpt_cond_latent IS NOT NULL AND xtts_embedding IS NOT NULL
        """).fetchall()
        self.xtts_model.speaker_manager.speakers = {}
        for row in speakers_query:
            firstname, surname, gpt_latent_list, gpt_shape_json, xtts_emb_list, xtts_shape_json = row
            speaker_name = f"{firstname} {surname}" if surname else firstname
            try:
                gpt_shape = tuple(json.loads(gpt_shape_json))
                xtts_shape = tuple(json.loads(xtts_shape_json))
                gpt_latent = torch.from_numpy(
                    np.array(gpt_latent_list, dtype=np.float32).reshape(gpt_shape)
                ).to(self.device)
                xtts_embedding = torch.from_numpy(
                    np.array(xtts_emb_list, dtype=np.float32).reshape(xtts_shape)
                ).to(self.device)
                self.xtts_model.speaker_manager.speakers[speaker_name] = {
                    "gpt_cond_latent": gpt_latent,
                    "speaker_embedding": xtts_embedding
                }
                logger.debug(f"Loaded {speaker_name} with shapes - GPT: {gpt_latent.shape}, Speaker: {xtts_embedding.shape}")
            except Exception as e:
                logger.error(f"Error loading speaker {speaker_name}: {e}")
                continue
        logger.info(f"Loaded {len(self.xtts_model.speaker_manager.speakers)} XTTS speakers from DuckDB native arrays.")

    def synthesize_stream_raw(self, text: str, speaker_name: str):
        """Performs raw XTTS inference and yields audio chunks as NumPy arrays."""
        speaker_data = self.xtts_model.speaker_manager.speakers.get(speaker_name)
        if speaker_data is None:
            raise ValueError(f"Speaker '{speaker_name}' not found.")
        gpt_cond_latent = speaker_data["gpt_cond_latent"].to(self.device)
        speaker_embedding = speaker_data["speaker_embedding"].to(self.device)
        for chunk in self.xtts_model.inference_stream(
            text=text,
            language=self.ctx.config["xtts_language"],
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            stream_chunk_size=self.ctx.config["xtts_stream_chunk_size"],
        ):
            yield chunk.cpu().numpy()


# =============================================================================
# SPEAKER RECOGNITION
# =============================================================================

class FastECAPASpeakerMatcher:
    """In-memory speaker matcher for ECAPA-TDNN embeddings with adaptive confidence scoring."""

    def __init__(self, ctx: ServerContext):
        self.ctx = ctx
        self.speaker_embeddings = {}
        self.embedding_matrix = None
        self.speaker_names = []
        self.speaker_uids = []
        self.load_embeddings_to_memory()

    def load_embeddings_to_memory(self):
        """Load all ECAPA embeddings from database into memory for fast comparison."""
        logger.info("Loading ECAPA embeddings into memory...")
        try:
            speakers_query = self.ctx.db.execute("""
                SELECT uid, firstname, surname, ecapa_embedding 
                FROM speakers 
                WHERE ecapa_embedding IS NOT NULL
            """).fetchall()
            if not speakers_query:
                logger.warning("No ECAPA embeddings found in database.")
                return
            embeddings_list = []
            for row in speakers_query:
                uid, firstname, surname, ecapa_embedding_list = row
                speaker_name = f"{firstname} {surname}" if surname else firstname
                try:
                    embedding_array = np.array(ecapa_embedding_list, dtype=np.float32)
                    embedding_normalized = embedding_array / np.linalg.norm(embedding_array)
                    self.speaker_embeddings[speaker_name] = embedding_normalized
                    embeddings_list.append(embedding_normalized)
                    self.speaker_names.append(speaker_name)
                    self.speaker_uids.append(uid)
                    logger.debug(f"Loaded ECAPA embedding for {speaker_name} (shape: {embedding_array.shape})")
                except Exception as e:
                    logger.error(f"Error loading ECAPA embedding for {speaker_name}: {e}")
                    continue
            if embeddings_list:
                self.embedding_matrix = np.vstack(embeddings_list)
                logger.debug(f"Created embedding matrix: {self.embedding_matrix.shape}")
                logger.info(f"Loaded {len(self.speaker_embeddings)} ECAPA embeddings into memory.")
            else:
                logger.warning("No valid ECAPA embeddings loaded.")
        except Exception as e:
            logger.error(f"Error loading ECAPA embeddings: {e}")

    def calculate_adaptive_confidence(self, best_score: float, second_score: float, domain_size: int) -> float:
        """Calculate confidence score that adapts based on domain size."""
        base_confidence = best_score
        gap = best_score - second_score
        gap_weight = min(0.4, 12.0 / domain_size)
        gap_bonus = gap * gap_weight
        composite = min(1.0, base_confidence + gap_bonus)
        return composite

    def find_best_match_with_nomatch_data(self, query_embedding, domain_size: Optional[int] = None) -> Tuple[Optional[str], Optional[int], float, Dict]:
        """Find the best matching speaker with additional data needed for nomatch scoring."""
        if self.embedding_matrix is None or len(self.speaker_embeddings) == 0:
            return None, None, 0.0, {"error": "No embeddings loaded"}
        if domain_size is None:
            domain_size = len(self.speaker_embeddings)
        try:
            if isinstance(query_embedding, torch.Tensor):
                query_array = query_embedding.detach().cpu().numpy()
            else:
                query_array = query_embedding
            query_normalized = query_array / np.linalg.norm(query_array)
            query_normalized = query_normalized.reshape(1, -1)
            similarities = cosine_similarity(query_normalized, self.embedding_matrix)[0]
            sorted_indices = np.argsort(similarities)[::-1]
            best_score = similarities[sorted_indices[0]]
            second_score = similarities[sorted_indices[1]] if len(similarities) > 1 else 0.0
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            median_similarity = np.median(similarities)
            best_speaker = self.speaker_names[sorted_indices[0]]
            best_uid = self.speaker_uids[sorted_indices[0]]
            confidence = self.calculate_adaptive_confidence(best_score, second_score, domain_size)
            nomatch_data = {
                "best_similarity": best_score,
                "second_similarity": second_score,
                "mean_similarity": mean_similarity,
                "std_similarity": std_similarity,
                "median_similarity": median_similarity,
                "similarity_gap": best_score - second_score,
                "domain_size": domain_size,
                "above_median_count": np.sum(similarities > median_similarity),
                "cosine_dissimilarity": 1.0 - best_score
            }
            return best_speaker, best_uid, confidence, nomatch_data
        except Exception as e:
            logger.error(f"Error in speaker matching: {e}")
            return None, None, 0.0, {"error": str(e)}

    def find_best_match(self, query_embedding, domain_size: Optional[int] = None) -> Tuple[Optional[str], Optional[int], float]:
        """Find the best matching speaker with adaptive confidence scoring."""
        if self.embedding_matrix is None or len(self.speaker_embeddings) == 0:
            return None, None, 0.0
        if domain_size is None:
            domain_size = len(self.speaker_embeddings)
        try:
            if isinstance(query_embedding, torch.Tensor):
                query_array = query_embedding.detach().cpu().numpy()
            else:
                query_array = query_embedding
            query_normalized = query_array / np.linalg.norm(query_array)
            query_normalized = query_normalized.reshape(1, -1)
            similarities = cosine_similarity(query_normalized, self.embedding_matrix)[0]
            sorted_indices = np.argsort(similarities)[::-1]
            best_score = similarities[sorted_indices[0]]
            second_score = similarities[sorted_indices[1]] if len(similarities) > 1 else 0.0
            best_speaker = self.speaker_names[sorted_indices[0]]
            best_uid = self.speaker_uids[sorted_indices[0]]
            confidence = self.calculate_adaptive_confidence(best_score, second_score, domain_size)
            return best_speaker, best_uid, confidence
        except Exception as e:
            logger.error(f"Error in speaker matching: {e}")
            return None, None, 0.0

    def find_best_match_with_details(self, query_embedding, domain_size: Optional[int] = None) -> Dict:
        """Find best match with detailed breakdown for debugging/analysis."""
        if self.embedding_matrix is None or len(self.speaker_embeddings) == 0:
            return {"speaker": None, "confidence": 0.0, "details": "No embeddings loaded"}
        if domain_size is None:
            domain_size = len(self.speaker_embeddings)
        try:
            if isinstance(query_embedding, torch.Tensor):
                query_array = query_embedding.detach().cpu().numpy()
            else:
                query_array = query_embedding
            query_normalized = query_array / np.linalg.norm(query_array)
            query_normalized = query_normalized.reshape(1, -1)
            similarities = cosine_similarity(query_normalized, self.embedding_matrix)[0]
            sorted_indices = np.argsort(similarities)[::-1]
            best_score = similarities[sorted_indices[0]]
            second_score = similarities[sorted_indices[1]] if len(similarities) > 1 else 0.0
            best_speaker = self.speaker_names[sorted_indices[0]]
            gap = best_score - second_score
            gap_weight = min(0.4, 12.0 / domain_size)
            gap_bonus = gap * gap_weight
            confidence = min(1.0, best_score + gap_bonus)
            return {
                "speaker": best_speaker,
                "confidence": confidence,
                "details": {
                    "raw_similarity": best_score,
                    "second_best_similarity": second_score,
                    "similarity_gap": gap,
                    "gap_weight": gap_weight,
                    "gap_bonus": gap_bonus,
                    "domain_size": domain_size,
                    "total_speakers_loaded": len(self.speaker_embeddings)
                }
            }
        except Exception as e:
            return {"speaker": None, "confidence": 0.0, "details": f"Error: {e}"}

    def get_speaker_count(self) -> int:
        return len(self.speaker_embeddings)

    def rebuild_embedding_matrix(self):
        """Rebuild embedding matrix from database."""
        logger.info("Rebuilding ECAPA embedding matrix from database...")
        self.speaker_embeddings.clear()
        self.speaker_names.clear()
        self.speaker_uids.clear()
        self.embedding_matrix = None
        self.load_embeddings_to_memory()

    def update_embedding_in_matrix(self, uid: int, speaker_name: str, new_embedding: np.ndarray) -> bool:
        """Update a specific speaker's embedding in the matrix without full rebuild."""
        try:
            if self.embedding_matrix is None or len(self.speaker_embeddings) == 0:
                logger.warning(f"[ECAPA Matcher] No embeddings loaded for {speaker_name}, performing full rebuild...")
                self.rebuild_embedding_matrix()
                return True
            try:
                speaker_idx = self.speaker_uids.index(uid)
                actual_speaker_name = self.speaker_names[speaker_idx]
            except ValueError:
                logger.info(f"[ECAPA Matcher] Speaker {speaker_name} (UID: {uid}) not found in matrix, adding new speaker...")
                return self.add_new_speaker_to_matrix(uid, speaker_name, new_embedding)
            embedding_normalized = new_embedding / np.linalg.norm(new_embedding)
            self.speaker_embeddings[actual_speaker_name] = embedding_normalized
            self.embedding_matrix[speaker_idx] = embedding_normalized
            logger.debug(f"[ECAPA Matcher] Updated embedding for {speaker_name} (UID: {uid}) in-place")
            return True
        except Exception as e:
            logger.error(f"[ECAPA Matcher] Error updating speaker embedding for {speaker_name} (UID: {uid}): {e}")
            return False

    def add_new_speaker_to_matrix(self, uid: int, speaker_name: str, new_embedding: np.ndarray) -> bool:
        """Add a completely new speaker to the matrix."""
        try:
            if uid in self.speaker_uids:
                existing_idx = self.speaker_uids.index(uid)
                existing_name = self.speaker_names[existing_idx]
                logger.warning(f"[ECAPA Matcher] Warning: UID {uid} already exists for {existing_name}, updating instead...")
                return self.update_embedding_in_matrix(uid, speaker_name, new_embedding)
            embedding_normalized = new_embedding / np.linalg.norm(new_embedding)
            self.speaker_embeddings[speaker_name] = embedding_normalized
            self.speaker_names.append(speaker_name)
            self.speaker_uids.append(uid)
            if self.embedding_matrix is None:
                self.embedding_matrix = embedding_normalized.reshape(1, -1)
            else:
                new_row = embedding_normalized.reshape(1, -1)
                self.embedding_matrix = np.vstack([self.embedding_matrix, new_row])
            logger.info(f"[ECAPA Matcher] Added new speaker {speaker_name} (UID: {uid}) to matrix (new size: {self.embedding_matrix.shape})")
            return True
        except Exception as e:
            logger.error(f"[ECAPA Matcher] Error adding new speaker {speaker_name} (UID: {uid}): {e}")
            return False

    def remove_speaker_from_matrix(self, uid: int, speaker_name: str = None) -> bool:
        """Remove a speaker from the matrix by UID."""
        try:
            try:
                speaker_idx = self.speaker_uids.index(uid)
                actual_speaker_name = self.speaker_names[speaker_idx]
            except ValueError:
                log_name = speaker_name if speaker_name else f"UID {uid}"
                logger.warning(f"[ECAPA Matcher] Speaker {log_name} not found for removal")
                return False
            del self.speaker_embeddings[actual_speaker_name]
            self.speaker_names.pop(speaker_idx)
            self.speaker_uids.pop(speaker_idx)
            if self.embedding_matrix.shape[0] == 1:
                self.embedding_matrix = None
            else:
                self.embedding_matrix = np.delete(self.embedding_matrix, speaker_idx, axis=0)
            log_name = speaker_name if speaker_name else actual_speaker_name
            logger.info(f"[ECAPA Matcher] Removed speaker {log_name} (UID: {uid}) from matrix")
            return True
        except Exception as e:
            log_name = speaker_name if speaker_name else f"UID {uid}"
            logger.error(f"[ECAPA Matcher] Error removing speaker {log_name}: {e}")
            return False


class ECAPASpeakerProcessor:
    """Unified ECAPA-TDNN speaker processor that handles both file-based extraction
    for initial speaker imprints and live audio buffer processing for real-time identification."""

    def __init__(self, ctx: ServerContext, model_path, device, ecapa_matcher, sample_rate=16000):
        logger.info("Pre-loading ECAPA-TDNN speaker processor...")
        self.ctx = ctx
        self.device = device
        self.model_path = model_path
        self.ecapa_matcher = ecapa_matcher
        self.sample_rate = sample_rate
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
            model_path, map_location=torch.device(self.device))
        self.model.eval()
        self.model.to(self.device)
        # Online processing state
        self.bytes_per_second = sample_rate * 2
        self.last_extraction_bytes = 0
        self.extraction_interval_bytes = self.bytes_per_second
        self.max_extractions = ctx.config["ecapa_max_extractions"]
        self.extraction_count = 0
        self.subsequent_nomatch = 0
        self.total_nomatch = 0
        self.nomatch_lower_threshold = ctx.config["ecapa_nomatch_lower_threshold"]
        self.nomatch_upper_threshold = ctx.config["ecapa_nomatch_upper_threshold"]
        self.UNCERTAIN_THRESHOLD = ctx.config["ecapa_uncertain_threshold"]
        self.CERTAIN_THRESHOLD = ctx.config["ecapa_certain_threshold"]
        logger.info("ECAPA-TDNN speaker processor loaded successfully")

    def extract_embedding_from_file(self, wav_path, sample_rate=None):
        """Extract ECAPA embedding from a WAV file."""
        if sample_rate is None:
            sample_rate = self.sample_rate
        try:
            logger.debug(f"[ECAPA] Extracting embedding from file: {wav_path}")
            with torch.no_grad():
                embeddings = self.model.get_embedding(wav_path)
            embedding_np = embeddings.cpu().numpy().squeeze()
            logger.debug(f"[ECAPA] Extracted embedding shape: {embedding_np.shape}")
            return embedding_np
        except Exception as e:
            logger.error(f"[ECAPA] Error extracting embedding from file {wav_path}: {e}")
            return None

    def extract_embedding_from_buffer(self, audio_int16, sample_rate=None):
        """Extract ECAPA embedding from int16 PCM audio buffer."""
        if sample_rate is None:
            sample_rate = self.sample_rate
        try:
            audio_float32 = AudioUtils.int16_to_float32(audio_int16)
            waveform = torch.from_numpy(audio_float32)
            if waveform.shape[0] > 1 if waveform.dim() > 1 else False:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            target_sr = 16000
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            device = next(self.model.parameters()).device
            waveform = waveform.to(device)
            with torch.no_grad():
                audio_length = torch.tensor([waveform.shape[1]], dtype=torch.long).to(device)
                _, embedding = self.model.forward(waveform, audio_length)
            embedding_np = embedding.cpu().numpy().squeeze()
            return embedding_np
        except Exception as e:
            logger.error(f"[ECAPA] Error extracting embedding from buffer: {e}")
            return None

    async def create_initial_speaker_imprint(self, wav_path, firstname, surname=None):
        """Extract both XTTS and ECAPA embeddings from a WAV file and store them."""
        wav_path = Path(wav_path)
        logger.info(f"[ECAPA] Creating initial speaker imprint for {firstname} {surname if surname else ''} from {wav_path}...")
        try:
            try:
                audio_duration = librosa.get_duration(path=str(wav_path))
                logger.debug(f"[ECAPA] Audio duration: {audio_duration:.2f} seconds")
            except Exception as e:
                logger.error(f"[ECAPA] Error getting audio duration: {e}")
                audio_duration = 0.0

            # Extract XTTS embeddings
            logger.debug("[ECAPA] Extracting XTTS embeddings...")
            with torch.no_grad():
                gpt_cond_latent, speaker_embedding = self.ctx.xtts.xtts_model.get_conditioning_latents(str(wav_path), 16000)
            logger.debug(f"[ECAPA] XTTS shapes - GPT: {gpt_cond_latent.shape}, Speaker: {speaker_embedding.shape}")
            gpt_latent_flat = gpt_cond_latent.cpu().numpy().flatten().tolist()
            xtts_embedding_flat = speaker_embedding.cpu().numpy().flatten().tolist()
            gpt_shape_json = json.dumps(list(gpt_cond_latent.shape))
            xtts_shape_json = json.dumps(list(speaker_embedding.shape))

            # Extract ECAPA embedding
            logger.debug("[ECAPA] Extracting ECAPA embedding...")
            ecapa_embedding = await asyncio.to_thread(self.extract_embedding_from_file, wav_path, self.sample_rate)
            if ecapa_embedding is None:
                logger.warning("[ECAPA] Failed to extract ECAPA embedding, storing XTTS data only")
                ecapa_embedding_flat = None
            else:
                ecapa_embedding_flat = ecapa_embedding.flatten().tolist()
                logger.debug(f"[ECAPA] ECAPA embedding shape: {ecapa_embedding.shape}")

            db = self.ctx.db
            def insert_speaker_data():
                db.execute("""
                    INSERT INTO speakers 
                    (firstname, surname, gpt_cond_latent, gpt_shape, xtts_embedding, xtts_shape, 
                    ecapa_embedding, total_duration_sec, sample_count, last_updated) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (firstname, surname, gpt_latent_flat, gpt_shape_json, xtts_embedding_flat,
                      xtts_shape_json, ecapa_embedding_flat, audio_duration, 1))

            await asyncio.to_thread(insert_speaker_data)
            logger.info(f"[ECAPA] Successfully stored complete speaker imprint for {firstname} in DuckDB")
            logger.debug(f"[ECAPA] Initial metadata - Duration: {audio_duration:.2f}s, Sample count: 1")

            # Reload speakers
            self.ctx.xtts._load_speakers_from_db()
            self.ecapa_matcher.rebuild_embedding_matrix()
            return True
        except Exception as e:
            logger.error(f"[ECAPA] Error creating speaker imprint for {firstname}: {e}")
            return False

    async def update_speaker_imprint_from_file(self, wav_path, uid):
        """Perform a cumulative update to an existing speaker's ECAPA embedding from file."""
        wav_path = Path(wav_path)
        logger.info(f"[ECAPA] Updating speaker imprint for UID {uid} with {wav_path}...")
        try:
            db = self.ctx.db
            def check_speaker_exists():
                return db.execute("""
                    SELECT uid, firstname, surname, ecapa_embedding, total_duration_sec, sample_count
                    FROM speakers WHERE uid = ?
                """, (uid,)).fetchone()

            existing_speaker = await asyncio.to_thread(check_speaker_exists)
            if existing_speaker is None:
                logger.error(f"[ECAPA] Speaker with UID {uid} does not exist in database")
                return False
            uid_db, firstname, surname, existing_embedding_list, total_duration, sample_count = existing_speaker
            speaker_name = f"{firstname} {surname}" if surname else firstname
            logger.debug(f"[ECAPA] Found existing speaker: {speaker_name}")
            try:
                new_audio_duration = librosa.get_duration(path=str(wav_path))
                logger.debug(f"[ECAPA] New audio duration: {new_audio_duration:.2f} seconds")
            except Exception as e:
                logger.error(f"[ECAPA] Error getting audio duration: {e}")
                return False
            new_embedding = await asyncio.to_thread(self.extract_embedding_from_file, wav_path, self.sample_rate)
            if new_embedding is None:
                logger.error(f"[ECAPA] Failed to extract ECAPA embedding from {wav_path}")
                return False
            logger.debug(f"[ECAPA] Successfully extracted new embedding shape: {new_embedding.shape}")
            if existing_embedding_list is not None and total_duration > 0:
                existing_embedding = np.array(existing_embedding_list, dtype=np.float32)
                existing_weight = total_duration / (total_duration + new_audio_duration)
                new_weight = new_audio_duration / (total_duration + new_audio_duration)
                combined_embedding = (existing_weight * existing_embedding + new_weight * new_embedding)
                logger.debug(f"[ECAPA] Combined embeddings - existing weight: {existing_weight:.3f}, new weight: {new_weight:.3f}")
            else:
                combined_embedding = new_embedding
                logger.debug("[ECAPA] No existing embedding data, using new embedding as baseline")

            def update_speaker_data():
                combined_embedding_list = combined_embedding.flatten().tolist()
                new_total_duration = (total_duration if total_duration else 0.0) + new_audio_duration
                new_sample_count = (sample_count if sample_count else 0) + 1
                db.execute("""
                    UPDATE speakers SET ecapa_embedding = ?, total_duration_sec = ?, sample_count = ?,
                        last_updated = CURRENT_TIMESTAMP WHERE uid = ?
                """, (combined_embedding_list, new_total_duration, new_sample_count, uid))
                return new_total_duration, new_sample_count

            new_total_duration, new_sample_count = await asyncio.to_thread(update_speaker_data)
            logger.info(f"[ECAPA] Successfully updated speaker {speaker_name} (UID: {uid})")
            logger.debug(f"[ECAPA] New totals - Duration: {new_total_duration:.2f}s, Samples: {new_sample_count}")
            logger.debug("[ECAPA] Rebuilding embedding matrix with updated data...")
            self.ecapa_matcher.rebuild_embedding_matrix()
            return True
        except Exception as e:
            logger.error(f"[ECAPA] Error updating speaker imprint for UID {uid}: {e}")
            return False

    async def update_speaker_imprint_from_buffer(self, uid, ecapa_embedding, audio_int16, sample_rate=None):
        """Perform a cumulative update to an existing speaker's ECAPA embedding using audio buffer data."""
        if sample_rate is None:
            sample_rate = self.sample_rate
        logger.info(f"[ECAPA] Updating speaker imprint for UID {uid} from audio buffer...")
        try:
            db = self.ctx.db
            def check_speaker_exists():
                return db.execute("""
                    SELECT uid, firstname, surname, ecapa_embedding, total_duration_sec, sample_count
                    FROM speakers WHERE uid = ?
                """, (uid,)).fetchone()

            existing_speaker = await asyncio.to_thread(check_speaker_exists)
            if existing_speaker is None:
                logger.error(f"[ECAPA] Speaker with UID {uid} does not exist in database")
                return False
            uid_db, firstname, surname, existing_embedding_list, total_duration, sample_count = existing_speaker
            speaker_name = f"{firstname} {surname}" if surname else firstname
            logger.debug(f"[ECAPA] Found existing speaker: {speaker_name}")
            try:
                audio_duration_samples = len(audio_int16)
                new_audio_duration = audio_duration_samples / sample_rate
                logger.debug(f"[ECAPA] Audio buffer duration: {new_audio_duration:.2f} seconds ({audio_duration_samples} samples at {sample_rate}Hz)")
            except Exception as e:
                logger.error(f"[ECAPA] Error calculating audio duration: {e}")
                return False
            if ecapa_embedding is None:
                logger.error("[ECAPA] Pre-computed ECAPA embedding is None")
                return False
            if not isinstance(ecapa_embedding, np.ndarray):
                logger.error(f"[ECAPA] ECAPA embedding must be a numpy array, got {type(ecapa_embedding)}")
                return False
            logger.debug(f"[ECAPA] Using pre-computed embedding shape: {ecapa_embedding.shape}")
            if existing_embedding_list is not None and total_duration > 0:
                existing_embedding = np.array(existing_embedding_list, dtype=np.float32)
                existing_weight = total_duration / (total_duration + new_audio_duration)
                new_weight = new_audio_duration / (total_duration + new_audio_duration)
                combined_embedding = (existing_weight * existing_embedding + new_weight * ecapa_embedding)
                logger.debug(f"[ECAPA] Combined embeddings - existing weight: {existing_weight:.3f}, new weight: {new_weight:.3f}")
            else:
                combined_embedding = ecapa_embedding
                logger.debug("[ECAPA] No existing embedding data, using new embedding as baseline")

            def update_speaker_data():
                combined_embedding_list = combined_embedding.flatten().tolist()
                new_total_duration = (total_duration if total_duration else 0.0) + new_audio_duration
                new_sample_count = (sample_count if sample_count else 0) + 1
                db.execute("""
                    UPDATE speakers SET ecapa_embedding = ?, total_duration_sec = ?, sample_count = ?,
                        last_updated = CURRENT_TIMESTAMP WHERE uid = ?
                """, (combined_embedding_list, new_total_duration, new_sample_count, uid))
                return new_total_duration, new_sample_count

            new_total_duration, new_sample_count = await asyncio.to_thread(update_speaker_data)
            logger.info(f"[ECAPA] Successfully updated speaker {speaker_name} (UID: {uid}) from buffer")
            logger.debug(f"[ECAPA] New totals - Duration: {new_total_duration:.2f}s, Samples: {new_sample_count}")
            update_success = self.ecapa_matcher.update_embedding_in_matrix(uid, speaker_name, combined_embedding)
            if not update_success:
                logger.warning("[ECAPA] Failed to update embedding matrix, falling back to full rebuild...")
                self.ecapa_matcher.rebuild_embedding_matrix()
            return True
        except Exception as e:
            logger.error(f"[ECAPA] Error updating speaker imprint for UID {uid} from buffer: {e}")
            return False

    def reset_for_new_utterance(self):
        """Reset the online processor state for a new utterance."""
        self.last_extraction_bytes = 0
        self.extraction_count = 0

    def should_extract_now(self, buffer_size_bytes):
        """Determine if we should extract an embedding based on buffer size."""
        if self.extraction_count >= self.max_extractions:
            return False
        bytes_since_last = buffer_size_bytes - self.last_extraction_bytes
        return bytes_since_last >= self.extraction_interval_bytes

    def calculate_nomatch_score(self, nomatch_data: Dict, buffer_duration: float) -> float:
        """Calculate the probability that the speaker is NOT in the database."""
        if "error" in nomatch_data:
            return 0.5
        base_dissimilarity = nomatch_data["cosine_dissimilarity"]
        min_duration = 0.8
        optimal_duration = 3.0
        if buffer_duration <= min_duration:
            duration_reliability = max(0.05, buffer_duration / min_duration * 0.3)
        elif buffer_duration >= optimal_duration:
            duration_reliability = 1.0
        else:
            progress = (buffer_duration - min_duration) / (optimal_duration - min_duration)
            duration_reliability = 0.3 + 0.7 * (progress ** 0.7)
        domain_size = nomatch_data["domain_size"]
        domain_adjustment = max(0.7, 1.0 - (0.3 * domain_size / (domain_size + 15.0)))
        mean_sim = nomatch_data["mean_similarity"]
        best_sim = nomatch_data["best_similarity"]
        std_sim = nomatch_data["std_similarity"]
        if std_sim > 0:
            z_score = (best_sim - mean_sim) / std_sim
            outlier_factor = max(0.0, -z_score * 0.1)
            outlier_factor = min(0.3, outlier_factor)
        else:
            outlier_factor = 0.0
        raw_nomatch_score = (base_dissimilarity * duration_reliability * domain_adjustment) + outlier_factor
        nomatch_score = min(1.0, max(0.0, raw_nomatch_score))
        return nomatch_score

    async def extract_and_match_from_buffer(self, audio_buffer, reason="scheduled"):
        """Extract ECAPA embedding from audio buffer and find best speaker match."""
        try:
            buffer_duration = len(audio_buffer) / self.bytes_per_second
            audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16)
            ecapa_embedding = await asyncio.to_thread(self.extract_embedding_from_buffer, audio_int16, self.sample_rate)
            if ecapa_embedding is None:
                return {"error": "Failed to extract embedding"}
            speaker_name, uid, confidence, nomatch_data = self.ecapa_matcher.find_best_match_with_nomatch_data(ecapa_embedding)
            nomatch_score = self.calculate_nomatch_score(nomatch_data, buffer_duration)

            # Cascading from confident nomatch to confident match
            if nomatch_score > self.nomatch_upper_threshold:
                speaker_result = "unregistered"
                uid_result = None
                speaker_confidence = "certain"
            elif nomatch_score > self.nomatch_lower_threshold:
                speaker_result = "unregistered(?)"
                uid_result = None
                speaker_confidence = "uncertain"
            elif confidence < self.UNCERTAIN_THRESHOLD:
                speaker_result = "unknown speaker"
                uid_result = None
                speaker_confidence = "uncertain"
            elif confidence < self.CERTAIN_THRESHOLD:
                speaker_result = f"{speaker_name}(?)"
                uid_result = uid
                speaker_confidence = "uncertain"
                self.subsequent_nomatch = 0
            elif confidence >= self.CERTAIN_THRESHOLD:
                speaker_result = f"{speaker_name}"
                uid_result = uid
                speaker_confidence = "certain"
                self.subsequent_nomatch = 0
                if reason == "silence":
                    try:
                        success = await self.update_speaker_imprint_from_buffer(uid, ecapa_embedding, audio_int16)
                        if success:
                            logger.debug(f"[ECAPA] Successfully updated imprint for {speaker_name}")
                        else:
                            logger.warning(f"[ECAPA] Failed to update imprint for {speaker_name}")
                    except Exception as e:
                        logger.error(f"[ECAPA] Error updating imprint: {e}")

            if speaker_confidence == "certain":
                logger.debug(f"[ECAPA] Speaker match result: {speaker_result} (confidence: {confidence:.3f}, {speaker_confidence})")

            if reason == "scheduled":
                self.extraction_count += 1
                self.last_extraction_bytes = len(audio_buffer)
                logger.debug(f"[ECAPA] nomatch score: {nomatch_score}")

            if reason == "silence" and nomatch_score >= self.nomatch_lower_threshold:
                self.subsequent_nomatch += 1
                self.total_nomatch += 1
                logger.debug(f"[ECAPA] Subsequent reliable nomatch count: {self.subsequent_nomatch}")
                logger.debug(f"[ECAPA] Total reliable nomatch count: {self.total_nomatch}")

            # Enrollment suggestion logic
            suggest_enrollment = False
            if reason == "silence" and nomatch_score >= self.nomatch_upper_threshold:
                logger.debug("[ECAPA] High nomatch confidence detected - consider triggering enrollment flow")
                self.subsequent_nomatch = 0
                self.total_nomatch = 0
                suggest_enrollment = True
            elif self.subsequent_nomatch >= 2:
                logger.debug("[ECAPA] 2 subsequent reasonable nomatch utterances - consider triggering enrollment flow")
                self.subsequent_nomatch = 0
                self.total_nomatch = 0
                suggest_enrollment = True
            elif self.total_nomatch >= 3:
                logger.debug("[ECAPA] 3 total reasonable nomatch utterances - consider triggering enrollment flow")
                self.subsequent_nomatch = 0
                self.total_nomatch = 0
                suggest_enrollment = True

            result = {
                "uid_result": uid_result,
                "confidence": confidence,
                "speaker_confidence": speaker_confidence,
                "speaker_result": speaker_result,
                "buffer_duration": buffer_duration,
                "extraction_reason": reason,
                "extraction_count": self.extraction_count + 1,
                "nomatch_score": nomatch_score,
                "nomatch_data": nomatch_data,
                "suggest_enrollment": suggest_enrollment
            }
            return result
        except Exception as e:
            logger.error(f"[ECAPA] Error in extract_and_match_from_buffer: {e}")
            return {"error": str(e)}


# =============================================================================
# RASA HANDLER - Communication + response processing
# =============================================================================

class RasaHandler:
    """Handles communication with Rasa server and processing of responses."""

    def __init__(self, ctx: ServerContext, rasa_url: str, timeout: int = 10):
        self.ctx = ctx
        self.rasa_url = rasa_url.rstrip('/')
        self.timeout = timeout
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def send_message(self, message, client_id, speaker_name=None, speaker_uid=None):
        """Send a message to Rasa and get the response."""
        if not self.session:
            logger.error("[Rasa] Session not initialized. Use async context manager.")
            return None
        try:
            payload = {"sender": client_id, "message": message}
            if speaker_name or speaker_uid:
                payload["metadata"] = {}
                if speaker_name:
                    payload["metadata"]["speaker_name"] = speaker_name
                if speaker_uid:
                    payload["metadata"]["speaker_uid"] = speaker_uid
            logger.debug(f"[Rasa] Sending message: '{message}'")
            async with self.session.post(f"{self.rasa_url}/webhooks/rest/webhook", json=payload) as response:
                if response.status == 200:
                    rasa_response = await response.json()
                    logger.debug(f"[Rasa] Received response: {rasa_response}")
                    return rasa_response
                else:
                    logger.error(f"[Rasa] HTTP Error {response.status}: {await response.text()}")
                    return None
        except asyncio.TimeoutError:
            logger.warning(f"[Rasa] Timeout after {self.timeout} seconds")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"[Rasa] Client error: {e}")
            return None
        except Exception as e:
            logger.error(f"[Rasa] Unexpected error: {e}")
            return None

    async def trigger_enrollment(self, client_id: str) -> bool:
        """Trigger enrollment flow by sending a system message."""
        if not self.session:
            logger.error("[Rasa] Session not initialized.")
            return False
        try:
            payload = {"sender": f"client_{client_id}", "message": "SYSTEM_TRIGGER_ENROLLMENT"}
            async with self.session.post(f"{self.rasa_url}/webhooks/rest/webhook", json=payload) as response:
                if response.status == 200:
                    rasa_response = await response.json()
                    logger.debug(f"[Rasa] Enrollment trigger response: {rasa_response}")
                    return await self.process_response(client_id, rasa_response)
                return False
        except Exception as e:
            logger.error(f"[Rasa] Error triggering enrollment: {e}")
            return False

    async def process_response(self, client_id: str, rasa_response: list) -> bool:
        """Process Rasa response and send appropriate messages/audio to client."""
        if not rasa_response:
            logger.warning("[Rasa] No response from Rasa")
            return False
        processed_any = False
        server_name = self.ctx.config.get("server_name", "Fawkes")
        for response_item in rasa_response:
            if "text" in response_item:
                response_text = response_item["text"]
                logger.debug(f"[Rasa] Processing text response: '{response_text}'")
                await self.ctx.msg.send_transcript(
                    client_id, server_name, "certain", "True", response_text, "certain")
                if not self.ctx.client_side_tts and self.ctx.active_websockets:
                    await self.ctx.client_queues[client_id]["tts_request_queue"].put(response_text)
                processed_any = True
            elif "custom" in response_item:
                custom_data = response_item["custom"]
                logger.debug(f"[Rasa] Processing custom response: {custom_data}")
                processed_any = True
        return processed_any

    async def handle_final_utterance(self, client_id, final_transcription_text, speaker_name, speaker_uid, speaker_confidence, nomatch_score):
        """Process final utterance through Rasa."""
        ecapa = self.ctx.ecapa_processor
        if not self.ctx.config.get("enable_rasa", False) or not self.session or not final_transcription_text.strip():
            return
        is_speaker_reliable = speaker_confidence >= ecapa.CERTAIN_THRESHOLD
        is_not_likely_nomatch = nomatch_score < ecapa.nomatch_upper_threshold
        is_reliable_utterance = (is_speaker_reliable and is_not_likely_nomatch)
        if is_reliable_utterance:
            logger.info(f"[Rasa] Reliable utterance detected, speaker name is: '{speaker_name}'")
        try:
            rasa_response = await self.send_message(
                final_transcription_text,
                client_id=f"client_{client_id}",
                speaker_name=speaker_name.removesuffix('(?)') if is_reliable_utterance else None,
                speaker_uid=speaker_uid
            )
            if rasa_response:
                success = await self.process_response(client_id, rasa_response)
                if not success:
                    logger.warning("[Rasa] No valid responses to process")
            else:
                logger.warning("[Rasa] Failed to get response from Rasa")
        except Exception as e:
            logger.error(f"[Rasa] Error in handle_final_utterance: {e}")


# =============================================================================
# API MODELS (Pydantic)
# =============================================================================

class EnrollmentAPIModels:
    """Pydantic models for speaker enrollment API endpoints."""

    class SpeakerQueryRequest(BaseModel):
        firstname: Optional[str] = None
        surname: Optional[str] = None
        full_name: Optional[str] = None
        @validator('full_name', 'firstname')
        def validate_query_params(cls, v, values):
            if not v and not values.get('firstname') and not values.get('full_name'):
                raise ValueError('Must provide either firstname/surname or full_name')
            return v

    class RecordPangramRequest(BaseModel):
        sender_id: str
        imprint_uid: Optional[str] = None
        imprint_firstname: Optional[str] = None
        imprint_surname: Optional[str] = None

    class EnrollmentStatusRequest(BaseModel):
        client_id: str
        status: str

    class SpeakerQueryResponse(BaseModel):
        uid: Optional[int] = None
        firstname: Optional[str] = None
        surname: Optional[str] = None
        confidence: Optional[float] = None
        status: Optional[str] = None
        success: bool = True

    class RecordPangramResponse(BaseModel):
        success: bool
        message: str = ""

    class EnrollmentStatusResponse(BaseModel):
        success: bool
        message: str = ""


class VoiceCloneAPIModels:
    class PassagesQueryRequest(BaseModel):
        action: str
        fuzzy_source: Optional[str] = None
        source_name: Optional[str] = None

    class PassagesQueryResponse(BaseModel):
        action: str
        sources: Optional[List[str]] = None
        source_name: Optional[str] = None
        confidence: Optional[float] = None
        quote: Optional[str] = None
        success: bool = True

    class VoiceCloneRequest(BaseModel):
        sender_id: str
        speaker: str
        quote: str

    class VoiceCloneResponse(BaseModel):
        success: bool
        message: str = ""


# =============================================================================
# ENROLLMENT TEXT UTILITIES
# =============================================================================

class EnrollmentTextUtils:
    """Utility functions for text processing during enrollment. All methods are static."""

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def is_utterance_on_topic(utterance: str, pangram_text: str, threshold: float = 0.3) -> bool:
        utterance_normalized = EnrollmentTextUtils.normalize_text(utterance)
        pangram_normalized = EnrollmentTextUtils.normalize_text(pangram_text)
        utterance_words = set(utterance_normalized.split())
        pangram_words = set(pangram_normalized.split())
        if not utterance_words:
            return False
        overlap = utterance_words & pangram_words
        overlap_ratio = len(overlap) / len(utterance_words)
        return overlap_ratio >= threshold

    @staticmethod
    def is_cancel_command(transcript: str) -> bool:
        normalized = transcript.lower().strip()
        cancel_patterns = [
            'cancel imprint', 'abort imprint', 'stop imprint',
            'cancel enrollment', 'abort enrollment', 'stop enrollment',
            'cancel recording', 'abort recording', 'stop recording'
        ]
        return any(pattern in normalized for pattern in cancel_patterns)

    @staticmethod
    def fuzzy_word_match(word1: str, word2: str) -> float:
        matcher = SequenceMatcher(None, word1.lower(), word2.lower())
        return matcher.ratio()

    @staticmethod
    def calculate_word_coverage(spoken_text: str, target_text: str,
                                fuzzy_threshold: float) -> tuple:
        spoken_normalized = EnrollmentTextUtils.normalize_text(spoken_text)
        target_normalized = EnrollmentTextUtils.normalize_text(target_text)
        spoken_words = spoken_normalized.split()
        target_words = target_normalized.split()
        matched_positions = {}
        matched_target_indices = set()
        for spoken_word in spoken_words:
            best_match_idx = None
            best_match_score = 0.0
            for i, target_word in enumerate(target_words):
                if i in matched_target_indices:
                    continue
                similarity = EnrollmentTextUtils.fuzzy_word_match(spoken_word, target_word)
                if similarity >= fuzzy_threshold and similarity > best_match_score:
                    best_match_score = similarity
                    best_match_idx = i
            if best_match_idx is not None:
                matched_positions[best_match_idx] = target_words[best_match_idx]
                matched_target_indices.add(best_match_idx)
        coverage_score = len(matched_positions) / len(target_words) if target_words else 0.0
        return coverage_score, matched_positions


# =============================================================================
# ENROLLMENT API HANDLER
# =============================================================================

class EnrollmentAPIHandler:
    """Handles FastAPI endpoints for speaker enrollment workflows."""

    def __init__(self, ctx: ServerContext, recording_manager, ecapa_matcher):
        self.ctx = ctx
        self.recording_manager = recording_manager
        self.ecapa_matcher = ecapa_matcher

    async def query_speaker(self, request: EnrollmentAPIModels.SpeakerQueryRequest) -> EnrollmentAPIModels.SpeakerQueryResponse:
        """Query speaker information using in-memory speaker matrix."""
        try:
            EXACT_MATCH_THRESHOLD = self.ctx.config["enrollment_exact_match_threshold"]
            if request.firstname:
                surname = request.surname or ""
                query_name = f"{request.firstname} {surname}".strip()
                if not self.ecapa_matcher.speaker_names:
                    logger.warning("[Enrollment API] No speakers in database")
                    return EnrollmentAPIModels.SpeakerQueryResponse(uid=None, success=True)
                best_match = None
                best_score = 0
                best_uid = None
                for speaker_name, uid in zip(self.ecapa_matcher.speaker_names, self.ecapa_matcher.speaker_uids):
                    score = SequenceMatcher(None, query_name.lower(), speaker_name.lower()).ratio()
                    if score > best_score:
                        best_score = score
                        best_match = speaker_name
                        best_uid = uid
                if best_score >= EXACT_MATCH_THRESHOLD:
                    fname, lname = best_match.split(' ', 1) if ' ' in best_match else (best_match, '')
                    logger.info(f"[Enrollment API] Exact match found: '{best_match}' (UID={best_uid}, score={best_score:.2f})")
                    return EnrollmentAPIModels.SpeakerQueryResponse(uid=best_uid, firstname=fname, surname=lname, success=True)
                else:
                    logger.info(f"[Enrollment API] No speakers matching '{query_name}' above {EXACT_MATCH_THRESHOLD*100}% threshold (best: {best_score:.2f})")
                    return EnrollmentAPIModels.SpeakerQueryResponse(uid=None, success=True)

            if request.full_name:
                if not self.ecapa_matcher.speaker_names:
                    logger.warning("[Enrollment API] No speakers in database")
                    return {'status': 'not_found', 'confidence': 0.0}
                is_firstname_only = ' ' not in request.full_name.strip()
                normalized_query = request.full_name.lower().replace('-', ' ').strip()
                if is_firstname_only:
                    logger.info(f"[Enrollment API] Detected firstname-only query: '{normalized_query}'")
                    firstname_matches = []
                    for speaker_name, uid in zip(self.ecapa_matcher.speaker_names, self.ecapa_matcher.speaker_uids):
                        fname = speaker_name.split(' ', 1)[0] if ' ' in speaker_name else speaker_name
                        normalized_fname = fname.lower().replace('-', ' ')
                        score = SequenceMatcher(None, normalized_query, normalized_fname).ratio()
                        firstname_matches.append({'score': score, 'firstname': fname, 'full_name': speaker_name, 'uid': uid})
                    firstname_matches.sort(key=lambda x: x['score'], reverse=True)
                    best_match = firstname_matches[0]
                    exact_matches = [m for m in firstname_matches if m['firstname'].lower() == normalized_query]
                    if exact_matches:
                        if len(exact_matches) == 1:
                            confidence = 1.0
                            best_match = exact_matches[0]
                            logger.info(f"[Enrollment API] Exact unique firstname match: '{best_match['firstname']}' -> confidence=1.0")
                        else:
                            num_matches = len(exact_matches)
                            confidence = 1.0 / num_matches
                            logger.info(f"[Enrollment API] Exact firstname match for {num_matches} people -> confidence={confidence:.2f}")
                    else:
                        if best_match['score'] >= 0.9:
                            close_matches = [m for m in firstname_matches if m['score'] >= 0.9]
                            if len(close_matches) == 1:
                                confidence = min(best_match['score'] * 1.1, 0.95)
                                logger.info(f"[Enrollment API] Unique close firstname match: '{best_match['firstname']}' -> confidence={confidence:.2f}")
                            else:
                                confidence = best_match['score']
                                logger.info(f"[Enrollment API] Multiple close matches -> confidence={confidence:.2f}")
                        else:
                            confidence = best_match['score']
                            logger.info(f"[Enrollment API] Fuzzy firstname match: '{best_match['firstname']}' -> confidence={confidence:.2f}")
                    fname, lname = best_match['full_name'].split(' ', 1) if ' ' in best_match['full_name'] else (best_match['full_name'], '')
                    logger.info(f"[Enrollment API] Firstname-only match result: '{fname} {lname}' (UID={best_match['uid']}, confidence={confidence:.2f})")
                    return {'status': 'success', 'firstname': fname, 'surname': lname, 'uid': best_match['uid'], 'confidence': round(confidence, 2)}
                else:
                    best_match = None
                    best_score = 0
                    best_uid = None
                    for speaker_name, uid in zip(self.ecapa_matcher.speaker_names, self.ecapa_matcher.speaker_uids):
                        normalized_name = speaker_name.lower().replace('-', ' ')
                        score = SequenceMatcher(None, normalized_query, normalized_name).ratio()
                        if score > best_score:
                            best_score = score
                            best_match = speaker_name
                            best_uid = uid
                    if best_match and best_score > 0:
                        fname, lname = best_match.split(' ', 1) if ' ' in best_match else (best_match, '')
                        logger.info(f"[Enrollment API] Full name fuzzy match for '{request.full_name}': '{best_match}' (UID={best_uid}, confidence={best_score:.2f})")
                        return {'status': 'success', 'firstname': fname, 'surname': lname, 'uid': best_uid, 'confidence': round(best_score, 2)}
                    logger.warning("[Enrollment API] No speakers in database")
                    return {'status': 'not_found', 'confidence': 0.0}

            logger.warning("[Enrollment API] Invalid query - no firstname or full_name provided")
            return EnrollmentAPIModels.SpeakerQueryResponse(uid=None, success=False)
        except Exception as e:
            logger.error(f"[Enrollment API] Error in query_speaker: {e}")
            logger.error(f"[Enrollment API] Request data: firstname={request.firstname}, surname={request.surname}, full_name={request.full_name}")
            return EnrollmentAPIModels.SpeakerQueryResponse(uid=None, success=False)

    async def record_pangram(self, request: EnrollmentAPIModels.RecordPangramRequest) -> EnrollmentAPIModels.RecordPangramResponse:
        """Initiate pangram recording for speaker enrollment."""
        try:
            client_id = request.sender_id.replace("client_", "") if request.sender_id.startswith("client_") else request.sender_id
            uid = int(request.imprint_uid) if request.imprint_uid else None
            firstname = request.imprint_firstname
            surname = request.imprint_surname
            logger.info(f"[Enrollment] Starting pangram recording for {firstname} {surname}, uid: {uid}")
            result = await self.recording_manager.start_recording(client_id=client_id, uid=uid, firstname=firstname, surname=surname)
            return EnrollmentAPIModels.RecordPangramResponse(success=True, message=f"Recording started: {result.get('pangram_text', '')}")
        except Exception as e:
            logger.error(f"[Enrollment] Error in record_pangram: {e}")
            return EnrollmentAPIModels.RecordPangramResponse(success=False, message=str(e))

    async def update_enrollment_status(self, request: EnrollmentAPIModels.EnrollmentStatusRequest) -> EnrollmentAPIModels.EnrollmentStatusResponse:
        """Update enrollment status for a client."""
        try:
            client_id = request.client_id
            status = request.status
            if client_id.startswith("client_"):
                client_id = client_id[7:]
            if client_id not in self.ctx.client_queues:
                return EnrollmentAPIModels.EnrollmentStatusResponse(success=False, message=f"Client {client_id} not found in active sessions")
            self.ctx.client_queues[client_id]["enrollment_active"] = False
            logger.info(f"[Enrollment API] Status updated for {client_id}: {status}")
            return EnrollmentAPIModels.EnrollmentStatusResponse(success=True, message=f"Enrollment status updated to {status}")
        except Exception as e:
            logger.error(f"[Enrollment API] Error updating enrollment status: {e}")
            return EnrollmentAPIModels.EnrollmentStatusResponse(success=False, message=str(e))


# =============================================================================
# ENROLLMENT RECORDING MANAGER
# =============================================================================

class EnrollmentRecordingManager:
    """Manages enrollment recording sessions. NO TIMING LOGIC - all timing handled in outer loop."""

    def __init__(self, ctx: ServerContext, ecapa_processor):
        self.ctx = ctx
        self.ecapa_processor = ecapa_processor
        self.active_sessions = {}
        self.server_name = ctx.config.get("server_name", "Fawkes")
        self.previous_score = {}
        self.decrease_count = {}

    async def start_recording(self, client_id: str, uid: Optional[int], firstname: Optional[str], surname: Optional[str]) -> Dict:
        """Start enrollment recording for a client."""
        cq = self.ctx.client_queues
        if client_id not in cq:
            return {"status": "error", "message": "Client not connected"}
        if "enrollment_state" in cq[client_id]:
            if cq[client_id]["enrollment_state"].get("recording_active"):
                return {"status": "error", "message": "Recording already active"}
        pangram_id, pangram_text = await self._select_pangram(uid)
        if pangram_id is None:
            return {"status": "error", "message": "No pangrams available"}
        cq[client_id]["enrollment_state"] = {
            "recording_active": True, "audio_buffer": [], "transcript_buffer": [],
            "pangram_id": pangram_id, "pangram_text": pangram_text, "uid": uid,
            "firstname": firstname, "surname": surname, "accumulated_transcript": "",
            "matched_positions": {}, "enrollment_last_speech_time": time.monotonic(),
            "off_topic_count": 0, "no_progress_count": 0
        }
        self.previous_score[client_id] = 0.0
        self.decrease_count[client_id] = 0
        logger.info(f"[Enrollment] Recording started for {client_id}")
        logger.info(f"[Enrollment] Pangram: {pangram_text}")
        await self.ctx.msg.send_transcript(client_id, self.server_name, "certain", "True", pangram_text, "certain")
        return {"status": "started", "pangram_id": pangram_id, "pangram_text": pangram_text}

    def get_frontend_highlight_data(self, client_id: str) -> dict:
        """Prepare matched word data for frontend highlighting."""
        cq = self.ctx.client_queues
        if client_id not in cq or "enrollment_state" not in cq[client_id]:
            return {}
        enrollment_state = cq[client_id]["enrollment_state"]
        pangram = enrollment_state['pangram_text']
        matched_positions = enrollment_state.get('matched_positions', {})
        normalized_pangram = EnrollmentTextUtils.normalize_text(pangram)
        pangram_words = normalized_pangram.split()
        return {
            "pangram_words": pangram_words,
            "matched_positions": list(matched_positions.keys()),
            "coverage_score": self.previous_score.get(client_id, 0.0)
        }

    async def process_utterance(self, client_id: str, utterance_audio: bytes, utterance_transcript: str) -> Optional[str]:
        """Process a completed utterance during enrollment recording."""
        cq = self.ctx.client_queues
        if client_id not in cq:
            return 'aborted'
        if "enrollment_state" not in cq[client_id]:
            return None
        enrollment_state = cq[client_id]["enrollment_state"]
        if not enrollment_state["recording_active"]:
            return None
        logger.info(f"[Enrollment] Processing utterance: '{utterance_transcript}'")
        audio_int16 = np.frombuffer(utterance_audio, dtype=np.int16)
        enrollment_state["audio_buffer"].append(audio_int16)
        if EnrollmentTextUtils.is_cancel_command(utterance_transcript):
            logger.info("[Enrollment] Cancel keyword detected")
            return await self.abort_recording(client_id, reason="cancel")
        pangram_text = enrollment_state["pangram_text"]
        if enrollment_state['accumulated_transcript']:
            enrollment_state['accumulated_transcript'] += " " + utterance_transcript
        else:
            enrollment_state['accumulated_transcript'] = utterance_transcript
        coverage_score, matched_positions = EnrollmentTextUtils.calculate_word_coverage(
            enrollment_state['accumulated_transcript'], pangram_text,
            self.ctx.config['enrollment_fuzzy_word_threshold'])
        enrollment_state['matched_positions'] = matched_positions
        logger.info(f"[Enrollment] Coverage score: {coverage_score:.1%}")
        prev_score = self.previous_score.get(client_id, 0.0)
        if coverage_score <= prev_score:
            enrollment_state["no_progress_count"] = enrollment_state.get("no_progress_count", 0) + 1
            logger.info(f"[Enrollment] No progress (count: {enrollment_state['no_progress_count']})")
            if enrollment_state["no_progress_count"] >= self.ctx.config['enrollment_max_decreases']:
                logger.warning("[Enrollment] Too many utterances without progress")
                return await self.abort_recording(client_id, "no_progress")
        else:
            enrollment_state["no_progress_count"] = 0
        if not EnrollmentTextUtils.is_utterance_on_topic(utterance_transcript, pangram_text, self.ctx.config['enrollment_off_topic_threshold']):
            enrollment_state["off_topic_count"] = enrollment_state.get("off_topic_count", 0) + 1
            logger.info(f"[Enrollment] Off-topic utterance (count: {enrollment_state['off_topic_count']})")
            if enrollment_state["off_topic_count"] >= self.ctx.config['enrollment_max_decreases']:
                logger.warning("[Enrollment] Too many off-topic utterances")
                return await self.abort_recording(client_id, "off_topic")
        self.previous_score[client_id] = coverage_score
        highlight_data = self.get_frontend_highlight_data(client_id)
        logger.debug(f"[Enrollment] Highlight data: {len(highlight_data.get('matched_positions', []))} words matched")
        if coverage_score >= self.ctx.config['enrollment_success_threshold']:
            logger.info(f"[Enrollment] Success threshold reached: {coverage_score:.1%}")
            return await self._complete_recording(client_id)
        return None

    async def abort_recording(self, client_id: str, reason: str = "other_speaker") -> str:
        """Abort recording (can be called publicly)."""
        logger.info(f"[Enrollment] ABORT called - client: {client_id}, reason: {reason}")
        messages = {
            "other_speaker": "Aborting imprint, please try again later with no other speakers present.",
            "timeout": "Aborting imprint, please try again later.",
            "no_progress": "Aborting imprint, please try again later.",
            "off_topic": "Aborting imprint, please try again later.",
            "cancel": "Aborting imprint, please try again later."
        }
        message = messages.get(reason, "Aborting imprint, please try again later.")
        return await self._abort_recording(client_id, message)

    async def _complete_recording(self, client_id: str) -> str:
        """Complete enrollment successfully."""
        cq = self.ctx.client_queues
        enrollment_state = cq[client_id]["enrollment_state"]
        try:
            wav_path = await self._save_audio_to_wav(client_id, enrollment_state)
            logger.info(f"[Enrollment] Saved: {wav_path}")
            uid = enrollment_state["uid"]
            firstname = enrollment_state["firstname"]
            surname = enrollment_state["surname"]
            if uid is None:
                success = await self.ecapa_processor.create_initial_speaker_imprint(
                    wav_path=str(wav_path), firstname=firstname, surname=surname)
                if success:
                    new_speaker = self.ctx.db.execute("""
                        SELECT uid FROM speakers WHERE firstname = ? AND surname = ? ORDER BY uid DESC LIMIT 1
                    """, [firstname, surname]).fetchone()
                    if new_speaker:
                        new_uid = new_speaker[0]
                        await self._mark_pangram_recited(new_uid, enrollment_state["pangram_id"])
            else:
                success = await self.ecapa_processor.update_speaker_imprint_from_file(wav_path=str(wav_path), uid=uid)
                if success:
                    await self._mark_pangram_recited(uid, enrollment_state["pangram_id"])
            message = "Enrollment completed successfully!"
            await self.ctx.msg.send_transcript(client_id, self.server_name, "certain", "True", message, "certain")
            if not self.ctx.client_side_tts and self.ctx.active_websockets:
                await cq[client_id]["tts_request_queue"].put(message)
            await self._notify_rasa(client_id, 'success')
            cq[client_id]["enrollment_active"] = False
            del cq[client_id]["enrollment_state"]
            self.previous_score.pop(client_id, None)
            self.decrease_count.pop(client_id, None)
            return 'success'
        except Exception as e:
            logger.error(f"[Enrollment] Error completing: {e}")
            return await self._abort_recording(client_id, "Enrollment failed.")

    async def _abort_recording(self, client_id: str, message: str) -> str:
        """Internal abort method."""
        cq = self.ctx.client_queues
        if client_id not in cq or "enrollment_state" not in cq[client_id]:
            return 'aborted'
        enrollment_state = cq[client_id]["enrollment_state"]
        try:
            if enrollment_state["audio_buffer"]:
                wav_path = await self._save_audio_to_wav(client_id, enrollment_state)
                logger.info(f"[Enrollment] Aborted, saved debug file: {wav_path}")
            await self.ctx.msg.send_transcript(client_id, self.server_name, "certain", "True", message, "certain")
            if not self.ctx.client_side_tts and self.ctx.active_websockets:
                await cq[client_id]["tts_request_queue"].put(message)
                logger.debug("Enqueued TTS message")
            await self._notify_rasa(client_id, 'aborted')
            cq[client_id]["enrollment_active"] = False
            del cq[client_id]["enrollment_state"]
            self.previous_score.pop(client_id, None)
            self.decrease_count.pop(client_id, None)
            return 'aborted'
        except Exception as e:
            logger.error(f"[Enrollment] Error during abort: {e}")
            if "enrollment_state" in cq[client_id]:
                del cq[client_id]["enrollment_state"]
            return 'aborted'

    async def _save_audio_to_wav(self, client_id: str, enrollment_state: Dict) -> Path:
        """Save audio buffer to WAV file."""
        session_id = client_id.replace('-', '')[:8]
        pangram_id = enrollment_state["pangram_id"]
        uid = enrollment_state["uid"]
        surname = enrollment_state.get("surname", "")
        firstname = enrollment_state.get("firstname", "")
        parts = [f"pangram{pangram_id}", session_id]
        if surname:
            parts.append(surname)
        if firstname:
            parts.append(firstname)
        if uid is not None:
            parts.append(f"uid{uid}")
        filename = "_".join(parts) + ".wav"
        wav_path = Path(self.ctx.config['samples_path']) / filename
        audio_buffer = enrollment_state["audio_buffer"]
        concatenated = np.concatenate(audio_buffer) if audio_buffer else np.array([], dtype=np.int16)
        with wave.open(str(wav_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(concatenated.tobytes())
        return wav_path

    async def _mark_pangram_recited(self, uid: int, pangram_id: int):
        """Mark pangram as recited in database."""
        try:
            result = self.ctx.db.execute("SELECT pangrams FROM speakers WHERE uid = ?", [uid]).fetchone()
            if result is None:
                return
            current = result[0] if result[0] else []
            if pangram_id not in current:
                current.append(pangram_id)
                self.ctx.db.execute("UPDATE speakers SET pangrams = ? WHERE uid = ?", [current, uid])
                logger.debug(f"[Enrollment] Marked pangram {pangram_id} for UID {uid}")
        except Exception as e:
            logger.error(f"[Enrollment] Error marking pangram: {e}")

    async def _notify_rasa(self, client_id: str, status: str) -> bool:
        """Notify Rasa of enrollment completion via RasaHandler."""
        system_messages = {
            "success": "SYSTEM_ENROLLMENT_SUCCESS",
            "aborted": "SYSTEM_ENROLLMENT_ABORT"
        }
        system_message = system_messages.get(status)
        if not system_message:
            logger.error(f"[Enrollment] Unknown notification status: '{status}'")
            return False
        if not self.ctx.rasa_handler or not self.ctx.rasa_handler.session:
            logger.warning("[Enrollment] RasaHandler not available for notification")
            return False
        try:
            response = await self.ctx.rasa_handler.send_message(
                system_message, client_id=f"client_{client_id}")
            if response:
                return await self.ctx.rasa_handler.process_response(client_id, response)
            return False
        except Exception as e:
            logger.error(f"[Enrollment] Error notifying Rasa: {e}")
            return False

    async def _select_pangram(self, uid: Optional[int]) -> Tuple[Optional[int], Optional[str]]:
        """Select an appropriate pangram for the speaker."""
        try:
            db = self.ctx.db
            if uid is not None:
                result = db.execute("SELECT pangrams FROM speakers WHERE uid = ?", [uid]).fetchone()
                recited = result[0] if (result and result[0]) else []
                if recited:
                    placeholders = ','.join('?' * len(recited))
                    query = f"SELECT id, text FROM pangrams WHERE id NOT IN ({placeholders}) ORDER BY RANDOM() LIMIT 1"
                    result = db.execute(query, recited).fetchone()
                else:
                    result = db.execute("SELECT id, text FROM pangrams ORDER BY RANDOM() LIMIT 1").fetchone()
                if not result:
                    logger.info("[Enrollment] All pangrams recited, selecting random pangram")
                    result = db.execute("SELECT id, text FROM pangrams ORDER BY RANDOM() LIMIT 1").fetchone()
            else:
                result = db.execute("SELECT id, text FROM pangrams ORDER BY RANDOM() LIMIT 1").fetchone()
            if result:
                return result[0], result[1]
            else:
                logger.warning("[Enrollment] No pangrams in database")
                return None, None
        except Exception as e:
            logger.error(f"[Enrollment] Error selecting pangram: {e}")
            return None, None


# =============================================================================
# VOICE CLONE API HANDLER
# =============================================================================

class VoiceCloneAPIHandler:
    """Handles FastAPI endpoints for voice cloning workflows."""

    def __init__(self, ctx: ServerContext):
        self.ctx = ctx

    async def query_passages(self, request: VoiceCloneAPIModels.PassagesQueryRequest) -> VoiceCloneAPIModels.PassagesQueryResponse:
        """Unified function to handle all passage queries."""
        try:
            db = self.ctx.db
            if request.action == "unique_sources":
                sources = db.execute("SELECT DISTINCT source FROM passages ORDER BY source").fetchall()
                source_list = [row[0] for row in sources]
                logger.debug(f"[Passages] Found {len(source_list)} unique sources")
                return VoiceCloneAPIModels.PassagesQueryResponse(action="unique_sources", success=True, sources=source_list)
            elif request.action == "match_source":
                if not request.fuzzy_source:
                    return VoiceCloneAPIModels.PassagesQueryResponse(action="match_source", success=False, source_name=None, confidence=0.0)
                sources = db.execute("SELECT DISTINCT source FROM passages").fetchall()
                available_sources = [row[0] for row in sources]
                if not available_sources:
                    return VoiceCloneAPIModels.PassagesQueryResponse(action="match_source", success=True, source_name=None, confidence=0.0)
                normalized_query = request.fuzzy_source.lower().replace('-', ' ')
                best_match = None
                best_score = 0.0
                for source in available_sources:
                    normalized_source = source.lower().replace('-', ' ')
                    score = SequenceMatcher(None, normalized_query, normalized_source).ratio()
                    if normalized_query in normalized_source or normalized_source in normalized_query:
                        score = max(score, self.ctx.config["passages_substring_boost"])
                    if score > best_score:
                        best_score = score
                        best_match = source
                logger.info(f"[Passages] Match: '{request.fuzzy_source}' -> '{best_match}' ({best_score:.2%})")
                return VoiceCloneAPIModels.PassagesQueryResponse(action="match_source", success=True, source_name=best_match, confidence=round(best_score, 2))
            elif request.action == "select_quote":
                if not request.source_name:
                    return VoiceCloneAPIModels.PassagesQueryResponse(action="select_quote", success=False, quote=None)
                quotes = db.execute("SELECT quote FROM passages WHERE source = ? ORDER BY RANDOM() LIMIT 1", (request.source_name,)).fetchall()
                if not quotes:
                    return VoiceCloneAPIModels.PassagesQueryResponse(action="select_quote", success=True, quote=None)
                selected_quote = quotes[0][0]
                logger.debug(f"[Passages] Selected quote from '{request.source_name}': {selected_quote[:60]}...")
                return VoiceCloneAPIModels.PassagesQueryResponse(action="select_quote", success=True, quote=selected_quote)
            else:
                logger.warning(f"[Passages] Invalid action: {request.action}")
                return VoiceCloneAPIModels.PassagesQueryResponse(action=request.action, success=False)
        except Exception as e:
            logger.error(f"[Passages] Error in query_passages: {e}")
            traceback.print_exc()
            return VoiceCloneAPIModels.PassagesQueryResponse(action=request.action, success=False)

    async def perform_voice_clone(self, request: VoiceCloneAPIModels.VoiceCloneRequest) -> VoiceCloneAPIModels.VoiceCloneResponse:
        """Perform voice cloning using the parallel TTS pipeline."""
        try:
            client_id = request.sender_id.replace("client_", "") if request.sender_id.startswith("client_") else request.sender_id
            speaker = request.speaker
            quote = request.quote
            cq = self.ctx.client_queues
            if client_id not in cq:
                logger.warning(f"[Voice Clone] Client {client_id} not found in active sessions")
                return VoiceCloneAPIModels.VoiceCloneResponse(success=False, message=f"Client {client_id} not found")
            if not self.ctx.active_websockets:
                logger.warning("[Voice Clone] No active websockets")
                return VoiceCloneAPIModels.VoiceCloneResponse(success=False, message="No active websocket connections")
            logger.info(f"[Voice Clone] Starting voice clone for speaker '{speaker}' with quote length {len(quote)}")
            server_name = self.ctx.config['server_name']
            await self.ctx.msg.send_transcript(client_id, speaker, "certain", "True", quote, "certain")
            tts = self.ctx.tts_manager
            if not self.ctx.client_side_tts and self.ctx.active_websockets:
                async def parallel_tts_pipeline():
                    try:
                        if client_id in cq:
                            cq[client_id]["tts_active"] = True
                            while not cq[client_id]["incoming_audio"].empty():
                                try:
                                    cq[client_id]["incoming_audio"].get_nowait()
                                except:
                                    break
                            logger.debug(f"[Voice Clone] Flushed audio buffer for {client_id}")
                        buffer = asyncio.Queue()
                        coqui_task = asyncio.create_task(tts.synthesize_xtts_to_buffer(speaker, quote, buffer))
                        await tts.stream_piper(client_id, "Compiling response, please wait a moment...")
                        await tts.stream_from_buffer(client_id, buffer)
                        await coqui_task
                        if client_id in cq:
                            while not cq[client_id]["outgoing_audio"].empty():
                                await asyncio.sleep(0.05)
                            await asyncio.sleep(0.2)
                            logger.debug(f"[Voice Clone] All audio chunks sent to client")
                            self.ctx.audio_playback_complete[client_id] = False
                            max_wait = self.ctx.config["voice_clone_playback_timeout"]
                            elapsed = 0
                            while not self.ctx.audio_playback_complete.get(client_id, False) and elapsed < max_wait:
                                await asyncio.sleep(0.1)
                                elapsed += 0.1
                            if self.ctx.audio_playback_complete.get(client_id, False):
                                logger.info(f"[Voice Clone] Client confirmed playback complete after {elapsed:.1f}s")
                            else:
                                logger.warning(f"[Voice Clone] Timeout waiting for playback complete after {max_wait}s, proceeding anyway")
                        logger.info("[Voice Clone] Completed successfully, notifying Rasa")
                        await self.notify_rasa_voice_clone_complete(client_id)
                    except Exception as e:
                        logger.error(f"[Voice Clone] Error in parallel TTS pipeline: {e}")
                        await self.notify_rasa_voice_clone_complete(client_id)
                    finally:
                        if client_id in cq:
                            cq[client_id]["tts_active"] = False
                await parallel_tts_pipeline()
            else:
                await self.notify_rasa_voice_clone_complete(client_id)
            return VoiceCloneAPIModels.VoiceCloneResponse(success=True, message=f"Voice clone started for speaker '{speaker}'")
        except Exception as e:
            logger.error(f"[Voice Clone] Error in perform_voice_clone: {e}")
            traceback.print_exc()
            return VoiceCloneAPIModels.VoiceCloneResponse(success=False, message=str(e))

    async def notify_rasa_voice_clone_complete(self, client_id: str) -> bool:
        """Notify Rasa that voice cloning is complete via RasaHandler."""
        if not self.ctx.rasa_handler or not self.ctx.rasa_handler.session:
            logger.warning("[Voice Clone] RasaHandler not available for notification")
            return False
        try:
            response = await self.ctx.rasa_handler.send_message(
                "SYSTEM_VOICE_CLONE_COMPLETE", client_id=f"client_{client_id}")
            if response:
                return await self.ctx.rasa_handler.process_response(client_id, response)
            return False
        except Exception as e:
            logger.error(f"[Voice Clone] Error notifying Rasa: {e}")
            return False


# =============================================================================
# TTS STREAM MANAGER
# =============================================================================

class TTSStreamManager:
    """Manages all TTS streaming operations."""

    def __init__(self, ctx: ServerContext):
        self.ctx = ctx

    async def process_queue(self, client_id):
        """Sequential TTS processor - ensures TTS messages play in order."""
        logger.info(f"[TTS Queue] Started processor for client {client_id}")
        cq = self.ctx.client_queues
        try:
            while True:
                if client_id not in cq:
                    logger.info(f"[TTS Queue] Client {client_id} disconnected, stopping processor")
                    break
                text = await cq[client_id]["tts_request_queue"].get()
                try:
                    if client_id in cq:
                        cq[client_id]["tts_active"] = True
                        while not cq[client_id]["incoming_audio"].empty():
                            try:
                                cq[client_id]["incoming_audio"].get_nowait()
                            except:
                                break
                    await self.stream_piper(client_id, text)
                    if client_id in cq:
                        while not cq[client_id]["outgoing_audio"].empty():
                            await asyncio.sleep(0.05)
                        await asyncio.sleep(0.2)
                    logger.debug(f"[TTS Queue] Completed: '{text[:50]}...' for {client_id}")
                except Exception as e:
                    logger.error(f"[TTS Queue] Error processing '{text[:50]}...': {e}")
                finally:
                    if client_id in cq:
                        cq[client_id]["tts_active"] = False
                cq[client_id]["tts_request_queue"].task_done()
        except Exception as e:
            logger.error(f"[TTS Queue] Processor error for {client_id}: {e}")
        finally:
            logger.info(f"[TTS Queue] Processor stopped for client {client_id}")

    async def stream_piper(self, client_id, text):
        """Stream Piper TTS audio to a client."""
        piper = self.ctx.piper_tts
        main_loop = self.ctx.main_loop
        cq = self.ctx.client_queues
        logger.debug(f"[TTS] Streaming (raw) to client {client_id}: {text}")
        if client_id not in cq:
            logger.warning(f"[TTS] Client {client_id} not in client_queues")
            return

        def blocking_piper_inference():
            try:
                piper_chunks_generator = piper.synthesize_stream_raw(text)
                for chunk in piper_chunks_generator:
                    converted_chunk = AudioUtils.prepare_for_streaming(chunk, 'raw', piper.sample_rate)
                    if converted_chunk:
                        asyncio.run_coroutine_threadsafe(
                            cq[client_id]["outgoing_audio"].put(converted_chunk), main_loop)
                    else:
                        pass
                asyncio.run_coroutine_threadsafe(cq[client_id]["outgoing_audio"].put(None), main_loop)
                logger.debug(f"[TTS] Finished streaming to client {client_id} (from thread).")
            except Exception as e:
                logger.error(f"[TTS] Error during Piper inference in thread for {client_id}: {e}")
                asyncio.run_coroutine_threadsafe(cq[client_id]["outgoing_audio"].put(None), main_loop)

        try:
            await asyncio.to_thread(blocking_piper_inference)
        except Exception as e:
            logger.error(f"[TTS] Error setting up Piper streaming to {client_id}: {e}")

    async def synthesize_xtts_to_buffer(self, speaker_name, text, buffer):
        """Kicks off XTTS inference in a background thread and puts audio chunks into buffer."""
        xtts = self.ctx.xtts
        main_loop = self.ctx.main_loop
        logger.debug(f"Computing XTTS for: {text}")

        def blocking_direct_inference():
            try:
                chunks_raw = xtts.synthesize_stream_raw(text, speaker_name)
                for chunk_np_float32 in chunks_raw:
                    if chunk_np_float32 is None or len(chunk_np_float32) == 0:
                        continue
                    converted_chunk = AudioUtils.prepare_for_streaming(chunk_np_float32, 'float32', xtts.sample_rate)
                    asyncio.run_coroutine_threadsafe(buffer.put(converted_chunk), main_loop)
                asyncio.run_coroutine_threadsafe(buffer.put(None), main_loop)
                logger.debug(f"[XTTS] Finished streaming to buffer.")
            except ValueError as ve:
                logger.error(f"[XTTS] Speaker Error during inference: {ve}")
                asyncio.run_coroutine_threadsafe(buffer.put(None), main_loop)
            except Exception as e:
                logger.error(f"[XTTS] General Error during inference: {e}")
                asyncio.run_coroutine_threadsafe(buffer.put(None), main_loop)

        await asyncio.to_thread(blocking_direct_inference)

    async def stream_from_buffer(self, client_id, buffer):
        """Streams pre-buffered audio chunks from a queue to a client's outgoing_audio queue."""
        logger.debug(f"[XTTS] Streaming from buffer queue to client {client_id}")
        cq = self.ctx.client_queues
        queue = cq.get(client_id, {}).get("outgoing_audio")
        if not queue:
            logger.warning(f"[XTTS] Client {client_id} not in client_queues or missing 'outgoing_audio'")
            return
        try:
            while True:
                chunk = await buffer.get()
                if chunk is None:
                    break
                await queue.put(chunk)
                await asyncio.sleep(0.015)
            await queue.put(None)
        except Exception as e:
            logger.error(f"[XTTS] Error streaming to {client_id}: {e}")


# =============================================================================
# WEBSOCKET MANAGER
# =============================================================================

class WebSocketManager:
    """Manages WebSocket client lifecycle and audio processing loop."""

    def __init__(self, ctx: ServerContext):
        self.ctx = ctx

    async def connection_handler(self, websocket):
        """Handle a new WebSocket connection."""
        client_id = str(uuid.uuid4())
        logger.info(f"New client connected: {client_id}")
        await self._websocket_session(websocket, client_id)

    async def _websocket_session(self, websocket, client_id):
        """Per-client task orchestrator."""
        self.ctx.active_websockets[client_id] = websocket
        self.ctx.client_queues[client_id] = {
            "incoming_audio": asyncio.Queue(),
            "outgoing_audio": asyncio.Queue(),
            "outgoing_text": asyncio.Queue(),
            "tts_request_queue": asyncio.Queue(),
            "tts_active": False,
        }
        try:
            incoming_task = asyncio.create_task(self._handle_incoming(websocket, client_id))
            outgoing_task = asyncio.create_task(self._handle_outgoing(websocket, client_id))
            asr_task = asyncio.create_task(self._process_audio_from_queue(client_id))
            tts_task = asyncio.create_task(self.ctx.tts_manager.process_queue(client_id))
            await asyncio.gather(incoming_task, outgoing_task, asr_task, tts_task)
        except asyncio.CancelledError:
            logger.debug(f"WebSocket task for {client_id} cancelled.")
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
        finally:
            logger.info(f"Cleaning up client {client_id}")
            self.ctx.client_queues.pop(client_id, None)
            self.ctx.active_websockets.pop(client_id, None)
            await websocket.close()

    async def _handle_incoming(self, websocket, client_id):
        """Handle incoming messages from a WebSocket client."""
        cq = self.ctx.client_queues
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await cq[client_id]["incoming_audio"].put(message)
                else:
                    logger.debug(f"Text message received: {message}")
                    if message == 'clientSideTTS':
                        self.ctx.client_side_tts = True
                        logger.info(f"Client has specified using client-side TTS.")
                    elif message == 'AUDIO_PLAYBACK_COMPLETE':
                        self.ctx.audio_playback_complete[client_id] = True
                        logger.debug(f"[Audio] Client {client_id} confirmed playback complete")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected.")
        finally:
            self.ctx.active_websockets.pop(client_id, None)
            self.ctx.client_queues.pop(client_id, None)

    async def _handle_outgoing(self, websocket, client_id):
        """Handle outgoing messages to a WebSocket client."""
        cq = self.ctx.client_queues
        try:
            while True:
                if client_id not in cq:
                    logger.debug(f"Client {client_id} no longer in client_queues. Exiting handle_outgoing.")
                    break
                if not cq[client_id]["outgoing_audio"].empty():
                    chunk = await cq[client_id]["outgoing_audio"].get()
                    if chunk is None:
                        await websocket.send(b"EOF")
                        continue
                    else:
                        await websocket.send(chunk)
                elif not cq[client_id]["outgoing_text"].empty():
                    text = await cq[client_id]["outgoing_text"].get()
                    await websocket.send(text)
                else:
                    await asyncio.sleep(0.01)
        except websockets.exceptions.ConnectionClosed:
            pass

    async def _process_audio_from_queue(self, client_id):
        """Main ASR/VAD/ECAPA processing loop."""
        cfg = self.ctx.config
        cq = self.ctx.client_queues
        nemo_transcriber = self.ctx.nemo_transcriber
        nemo_vad = self.ctx.nemo_vad
        canary_qwen = self.ctx.canary_qwen
        ecapa_processor = self.ctx.ecapa_processor
        enrollment_manager = self.ctx.enrollment_manager
        rasa_handler = self.ctx.rasa_handler
        msg = self.ctx.msg

        chunk_size_ms = cfg["nemo_lookahead_size"] + cfg["nemo_encoder_step_length"]
        bytes_per_chunk = int(cfg["audio_sample_rate"] * chunk_size_ms / 1000) * 2
        audio_buffer = b''
        current_utterance_buffer = b''
        is_speaking = False
        silence_counter = 0
        last_utterance_time = None
        last_speech_time = None
        last_prompt_time = None
        SILENCE_CHUNKS_THRESHOLD = cfg["silence_chunks_threshold"]
        final_transcription_text = ""
        SPEAKER = cfg["default_speaker"]
        SPEAKER_CONFIDENCE = cfg["default_speaker_confidence"]
        ASR_CONFIDENCE = cfg["default_asr_confidence"]
        server_name = cfg.get("server_name", "Fawkes")
        speaker_uid = None
        confidence = 0.0
        nomatch_score = 0.0

        try:
            while True:
                try:
                    audio_data = await cq[client_id]["incoming_audio"].get()
                    if audio_data is None or len(audio_data) == 0:
                        await asyncio.sleep(0.001)
                        continue
                    if client_id in cq and cq[client_id].get("tts_active", False):
                        continue
                    audio_buffer += audio_data
                    while len(audio_buffer) >= bytes_per_chunk:
                        chunk_bytes = audio_buffer[:bytes_per_chunk]
                        audio_buffer = audio_buffer[bytes_per_chunk:]
                        audio_chunk_np = np.frombuffer(chunk_bytes, dtype=np.int16)
                        if audio_chunk_np.ndim != 1:
                            audio_chunk_np = audio_chunk_np.squeeze()

                        is_voice_active_in_chunk = await asyncio.to_thread(nemo_vad.detect_voice, audio_chunk_np)

                        if is_voice_active_in_chunk:
                            current_utterance_buffer += chunk_bytes
                            silence_counter = 0
                            last_utterance_time = time.monotonic()

                            if not is_speaking:
                                is_speaking = True
                                nemo_transcriber.previous_hypotheses = None
                                nemo_transcriber.pred_out_stream = None
                                nemo_transcriber.step_num = 0
                                num_channels = nemo_transcriber.asr_model.cfg.preprocessor.features
                                nemo_transcriber.cache_pre_encode = torch.zeros(
                                    (1, num_channels, nemo_transcriber.pre_encode_cache_size),
                                    device=nemo_transcriber.device)
                                nemo_transcriber.cache_last_channel, nemo_transcriber.cache_last_time, nemo_transcriber.cache_last_channel_len = \
                                    nemo_transcriber.asr_model.encoder.get_initial_cache_state(batch_size=1)
                                text = ""
                                final_transcription_text = ""
                                ecapa_processor.reset_for_new_utterance()

                            text = await asyncio.to_thread(nemo_transcriber.transcribe_chunk, audio_chunk_np)
                            final_transcription_text = text
                            if final_transcription_text != "":
                                last_speech_time = time.monotonic()
                                if "enrollment_state" in cq[client_id]:
                                    if cq[client_id]["enrollment_state"].get("recording_active"):
                                        cq[client_id]["enrollment_state"]["enrollment_last_speech_time"] = last_speech_time

                            if ecapa_processor.should_extract_now(len(current_utterance_buffer)):
                                ecapa_result = await ecapa_processor.extract_and_match_from_buffer(
                                    current_utterance_buffer, reason="scheduled")
                                if "error" not in ecapa_result:
                                    logger.info(f"[Speaker ID] {ecapa_result['speaker_result']}")
                                    SPEAKER = ecapa_result['speaker_result']
                                    SPEAKER_CONFIDENCE = ecapa_result['speaker_confidence']
                                    speaker_uid = ecapa_result['uid_result']

                            data_to_send = {
                                "speaker": SPEAKER, "speaker_confidence": SPEAKER_CONFIDENCE,
                                "final": False, "transcript": text, "asr_confidence": ASR_CONFIDENCE
                            }
                            await msg.send_to_client(client_id, json.dumps(data_to_send))

                            # ABORT recording if speaker is identified 'certain' different from imprint speaker
                            if "enrollment_state" in cq[client_id]:
                                if cq[client_id]["enrollment_state"]["recording_active"]:
                                    enrollment_state = cq[client_id]["enrollment_state"]
                                    expected_uid = enrollment_state["uid"]
                                    detected_uid = speaker_uid
                                    if expected_uid is None and detected_uid is not None and SPEAKER_CONFIDENCE == "certain":
                                        logger.warning(f"[Enrollment] ABORT: Other speaker detected (UID {detected_uid})")
                                        await enrollment_manager.abort_recording(client_id=client_id, reason="other_speaker")
                                    elif expected_uid is not None and detected_uid is not None and detected_uid != expected_uid and SPEAKER_CONFIDENCE == "certain":
                                        logger.warning(f"[Enrollment] ABORT: Wrong speaker (expected {expected_uid}, got {detected_uid})")
                                        await enrollment_manager.abort_recording(client_id=client_id, reason="other_speaker")

                        else:  # VAD indicates silence
                            silence_counter += 1
                            current_time = time.monotonic()

                            # Enrollment timeout/reminder logic
                            if "enrollment_state" in cq[client_id]:
                                enrollment_state = cq[client_id]["enrollment_state"]
                                if enrollment_state["recording_active"]:
                                    enrollment_last_speech = enrollment_state.get("enrollment_last_speech_time")
                                    nonspeech_duration = (current_time - enrollment_last_speech) if enrollment_last_speech else 0.0
                                    if nonspeech_duration >= cfg['enrollment_timeout']:
                                        logger.debug(f"silence duration = {nonspeech_duration}")
                                        await enrollment_manager.abort_recording(client_id, reason="timeout")
                                    elif nonspeech_duration >= cfg['enrollment_reminder_interval']:
                                        if last_prompt_time is None or (current_time - last_prompt_time) >= cfg['enrollment_reminder_interval']:
                                            reminder_text = "PLEASE FINISH RECITING THE PROMPT"
                                            await msg.send_transcript(client_id, server_name, "certain", "True", reminder_text, "certain")
                                            last_prompt_time = current_time

                            if is_speaking and silence_counter >= SILENCE_CHUNKS_THRESHOLD:
                                logger.debug("Acoustic finality detected. Processing full utterance with offline model...")

                                if len(current_utterance_buffer) > 0:
                                    final_ecapa_result = await ecapa_processor.extract_and_match_from_buffer(
                                        current_utterance_buffer, reason="silence")
                                    if "error" not in final_ecapa_result:
                                        logger.info(f"[Final Speaker ID] {final_ecapa_result['speaker_result']}")
                                        SPEAKER = final_ecapa_result['speaker_result']
                                        SPEAKER_CONFIDENCE = final_ecapa_result['speaker_confidence']
                                        nomatch_score = final_ecapa_result['nomatch_score']
                                        confidence = final_ecapa_result['confidence']
                                        speaker_uid = final_ecapa_result['uid_result']
                                
                                audio_int16 = np.frombuffer(current_utterance_buffer, dtype=np.int16)

                                final_canary_transcription = await asyncio.to_thread(
                                    canary_qwen.transcribe_final, audio_int16, cfg["audio_sample_rate"])
                                logger.info(f"Canary-Qwen final transcription: '{final_canary_transcription}'")

                                if final_canary_transcription.strip():
                                    final_transcription_text = final_canary_transcription
                                
                                if final_transcription_text:
                                    data_to_send = {
                                        "speaker": SPEAKER, "speaker_confidence": SPEAKER_CONFIDENCE,
                                        "final": True, "transcript": final_transcription_text,
                                        "asr_confidence": ASR_CONFIDENCE
                                    }
                                    await msg.send_to_client(client_id, json.dumps(data_to_send))

                                    if "enrollment_state" in cq[client_id] and cq[client_id]["enrollment_state"]["recording_active"]:
                                        result = await enrollment_manager.process_utterance(
                                            client_id=client_id, utterance_audio=current_utterance_buffer,
                                            utterance_transcript=final_transcription_text)
                                        if result in ['success', 'aborted']:
                                            logger.info(f"[Enrollment] Recording {result}")
                                    else:
                                        await rasa_handler.handle_final_utterance(
                                            client_id, final_transcription_text, SPEAKER, speaker_uid, confidence, nomatch_score)

                                if "suggest_enrollment" in final_ecapa_result and final_ecapa_result["suggest_enrollment"]:
                                    if not cq[client_id].get("enrollment_active", False):
                                        recording_active = (
                                            "enrollment_state" in cq[client_id]
                                            and cq[client_id]["enrollment_state"]["recording_active"]
                                        )
                                        if not recording_active:
                                            cq[client_id]["enrollment_active"] = True
                                            logger.info(f"[ECAPA] Triggering enrollment flow for {client_id}")
                                            if rasa_handler and rasa_handler.session:
                                                await rasa_handler.trigger_enrollment(client_id)

                                # Reset for next utterance
                                is_speaking = False
                                silence_counter = 0
                                current_utterance_buffer = b''
                                text = ""
                                SPEAKER = cfg["default_speaker"]
                                SPEAKER_CONFIDENCE = cfg["default_speaker_confidence"]
                                ecapa_processor.reset_for_new_utterance()

                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.01)
                except Exception as e:
                    logger.error(f"Error processing audio for {client_id}: {e}")
                    break
                finally:
                    pass
        finally:
            logger.info("Async Audio processing stopped")


# =============================================================================
# UTILITY FUNCTIONS (testing / maintenance)
# =============================================================================

async def save_utterance_async(audio_bytes: bytes) -> str:
    """Saves a complete audio utterance to a WAV file in a non-blocking manner."""
    OUTPUT_DIR = "utterances"
    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2
    NUM_CHANNELS = 1
    filename = f"utterance_{uuid.uuid4()}.wav"
    filepath = f"{OUTPUT_DIR}/{filename}"

    def _save_file():
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(NUM_CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_bytes)
            logger.info(f"Successfully saved utterance to {filepath}")
        except Exception as e:
            logger.error(f"Error saving utterance to {filepath}: {e}")
            raise e

    await asyncio.to_thread(_save_file)
    return filepath


async def manual_sequential_ecapa(ctx, firstname, surname, update_wav_paths):
    """Perform sequential ECAPA embedding updates for a specific speaker (testing utility)."""
    logger.info(f"--- Sequential ECAPA updates for {firstname} {surname if surname else ''} ---")
    db = ctx.db
    if surname:
        uid_result = db.execute("SELECT uid FROM speakers WHERE firstname = ? AND surname = ?", (firstname, surname)).fetchone()
        speaker_display_name = f"{firstname} {surname}"
    else:
        uid_result = db.execute("SELECT uid FROM speakers WHERE firstname = ? AND surname IS NULL", (firstname,)).fetchone()
        speaker_display_name = firstname
    if not uid_result:
        error_msg = f"Error: Could not find speaker '{speaker_display_name}' in database"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "speaker": speaker_display_name, "updates_attempted": 0, "updates_successful": 0}
    speaker_uid = uid_result[0]
    logger.info(f"Found {speaker_display_name} with UID: {speaker_uid}")
    initial_metadata = db.execute("SELECT total_duration_sec, sample_count FROM speakers WHERE uid = ?", (speaker_uid,)).fetchone()
    initial_duration, initial_count = initial_metadata if initial_metadata else (0.0, 0)
    logger.info(f"Initial state - Duration: {initial_duration:.2f}s, Sample count: {initial_count}")
    successful_updates = 0
    failed_updates = []
    try:
        for i, wav_path in enumerate(update_wav_paths, 1):
            logger.info(f"Processing update {i}/{len(update_wav_paths)}: {Path(wav_path).name}")
            try:
                success = await ctx.ecapa_processor.update_speaker_imprint_from_file(wav_path, speaker_uid)
                if success:
                    successful_updates += 1
                    logger.info(f"Update {i} completed successfully")
                else:
                    failed_updates.append({"index": i, "path": wav_path, "error": "Function returned False"})
                    logger.warning(f"Update {i} failed - function returned False")
            except FileNotFoundError:
                failed_updates.append({"index": i, "path": wav_path, "error": "File not found"})
                logger.error(f"Update {i} failed - file not found: {wav_path}")
            except Exception as e:
                failed_updates.append({"index": i, "path": wav_path, "error": str(e)})
                logger.error(f"Update {i} failed - error: {e}")
    except Exception as e:
        error_msg = f"Critical error during sequential updates: {e}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "speaker": speaker_display_name,
                "updates_attempted": len(update_wav_paths), "updates_successful": successful_updates, "failed_updates": failed_updates}
    final_metadata = db.execute("SELECT total_duration_sec, sample_count, last_updated FROM speakers WHERE uid = ?", (speaker_uid,)).fetchone()
    if final_metadata:
        final_duration, final_count, last_update = final_metadata
        logger.info(f"--- Final metadata for {speaker_display_name} ---")
        logger.info(f"Total duration: {final_duration:.2f}s (+{final_duration - initial_duration:.2f}s)")
        logger.info(f"Sample count: {final_count} (+{final_count - initial_count})")
        logger.info(f"Last updated: {last_update}")
    logger.info(f"--- Update Summary ---")
    logger.info(f"Speaker: {speaker_display_name}")
    logger.info(f"Updates attempted: {len(update_wav_paths)}")
    logger.info(f"Updates successful: {successful_updates}")
    logger.info(f"Updates failed: {len(failed_updates)}")
    return {"success": successful_updates > 0, "speaker": speaker_display_name, "speaker_uid": speaker_uid,
            "updates_attempted": len(update_wav_paths), "updates_successful": successful_updates,
            "updates_failed": len(failed_updates), "failed_updates": failed_updates}


# =============================================================================
# FASTAPI ROUTE REGISTRATION
# =============================================================================

# Module-level ctx reference for FastAPI routes (set during main())
_ctx: ServerContext = None

app = FastAPI()

@app.post("/api/speakers/query", response_model=EnrollmentAPIModels.SpeakerQueryResponse)
async def query_speaker_endpoint(request: EnrollmentAPIModels.SpeakerQueryRequest):
    return await _ctx.enrollment_api.query_speaker(request)

@app.post("/api/passages/query", response_model=VoiceCloneAPIModels.PassagesQueryResponse)
async def query_passages_endpoint(request: VoiceCloneAPIModels.PassagesQueryRequest):
    return await _ctx.voiceclone_api.query_passages(request)

@app.post("/api/enrollment/record", response_model=EnrollmentAPIModels.RecordPangramResponse)
async def record_pangram_endpoint(request: EnrollmentAPIModels.RecordPangramRequest):
    return await _ctx.enrollment_api.record_pangram(request)

@app.post("/api/enrollment/status", response_model=EnrollmentAPIModels.EnrollmentStatusResponse)
async def update_enrollment_status_endpoint(request: EnrollmentAPIModels.EnrollmentStatusRequest):
    return await _ctx.enrollment_api.update_enrollment_status(request)

@app.post("/api/voice-clone", response_model=VoiceCloneAPIModels.VoiceCloneResponse)
async def voice_clone_endpoint(request: VoiceCloneAPIModels.VoiceCloneRequest):
    return await _ctx.voiceclone_api.perform_voice_clone(request)


# =============================================================================
# MAIN - Application entry point
# =============================================================================

async def main():
    global _ctx

    ctx = ServerContext(CONFIG)
    _ctx = ctx
    ctx.main_loop = asyncio.get_event_loop()
    ctx.fastapi_app = app

    # Database
    db_manager = DatabaseManager(CONFIG["duckdb_path"])
    db_manager.setup_tables()
    ctx.db = db_manager

    # Message router
    ctx.msg = MessageRouter(ctx)

    # Models
    ctx.piper_tts = PiperTTS(CONFIG["piper_model_path"])

    ctx.xtts = XTTSWrapper(
        ctx, CONFIG["xtts_model_dir"], CONFIG["inference_device"], CONFIG["speakers_dir"])

    ctx.nemo_vad = NemoVAD(
        model_path=CONFIG["nemo_vad_model_path"],
        device=CONFIG["inference_device"],
        sample_rate=CONFIG["vad_sample_rate"])

    ctx.nemo_transcriber = NemoStreamingTranscriber(
        model_path=CONFIG["nemo_model_path"],
        decoder_type=CONFIG["nemo_decoder_type"],
        lookahead_size=CONFIG["nemo_lookahead_size"],
        encoder_step_length=CONFIG["nemo_encoder_step_length"],
        device=CONFIG["inference_device"],
        sample_rate=CONFIG["audio_sample_rate"])

    ctx.canary_qwen = None
    ctx.canary_qwen = CanaryQwenTranscriber(
        model_path=CONFIG["canary_qwen_model_path"],
        device=CONFIG["inference_device"],
        max_new_tokens=CONFIG["canary_max_new_tokens"])
    

    ctx.ecapa_matcher = FastECAPASpeakerMatcher(ctx)

    ctx.ecapa_processor = ECAPASpeakerProcessor(
        ctx, model_path=CONFIG["ecapa_tdnn_model_path"],
        device=CONFIG["inference_device"],
        ecapa_matcher=ctx.ecapa_matcher,
        sample_rate=CONFIG["audio_sample_rate"])

    # Handlers
    ctx.enrollment_manager = EnrollmentRecordingManager(ctx, ctx.ecapa_processor)

    ctx.enrollment_api = EnrollmentAPIHandler(ctx, ctx.enrollment_manager, ctx.ecapa_matcher)

    ctx.voiceclone_api = VoiceCloneAPIHandler(ctx)

    ctx.tts_manager = TTSStreamManager(ctx)

    ctx.ws_manager = WebSocketManager(ctx)

    # Rasa
    if CONFIG.get("enable_rasa", False):
        ctx.rasa_handler = RasaHandler(ctx, CONFIG["rasa_url"], CONFIG["rasa_timeout"])
        await ctx.rasa_handler.__aenter__()
        logger.info("[Rasa] Client initialized and connected")
    else:
        ctx.rasa_handler = None

    # Start FastAPI server
    fastapi_config = uvicorn.Config(app, host=CONFIG['fastapi_host'], port=CONFIG['fastapi_port'], log_level="info")
    fastapi_server = uvicorn.Server(fastapi_config)
    fastapi_task = asyncio.create_task(fastapi_server.serve())

    # Start WebSocket server
    websocket_server = await websockets.serve(
        ctx.ws_manager.connection_handler, CONFIG['websocket_host'], CONFIG['websocket_port'])

    try:
        logger.info(f"Starting FastAPI server on http://{CONFIG['fastapi_host']}:{CONFIG['fastapi_port']}")
        logger.info(f"Starting WebSocket server on ws://{CONFIG['websocket_host']}:{CONFIG['websocket_port']}")
        await asyncio.gather(fastapi_task, websocket_server.wait_closed())
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Cleaning up resources...")
        if ctx.rasa_handler:
            try:
                await ctx.rasa_handler.__aexit__(None, None, None)
                logger.info("[Rasa] Client connection closed")
            except Exception as e:
                logger.error(f"[Rasa] Error during cleanup: {e}")
        websocket_server.close()
        await websocket_server.wait_closed()
        logger.info("[WebSocket] Server closed")
        if ctx.db:
            ctx.db.close()
            logger.info("[Database] Connection closed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
