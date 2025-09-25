# Adding Rasa integration

# CLEANUP IDEAS
# move imports to required models loading class(?)
# //rename classes send_message_to_clients() to send_message_to_client()
# rename "NeMo" model's classes/functions for more explicit(?)
# //Write utterances to files in non-blocking manner for testing
# NOTES
# Canary is working but GPU memory paging even with other models diabled. Need more VRAM
# May circle back for Lexical/Audio P&C using NeMo's models (Canary is underperforming)
# May also experiment queueing multiple utterances in a floating frame to increase context available to Canary-Qwen/Samba
# Samba-ASR adoption may require language model fusion/contextual rescoring/full LLM integration/P&C
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
from scipy.io.wavfile import write as wav_write
import duckdb
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional, List, Dict, Any, BinaryIO
import torchaudio
import librosa
import aiohttp

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
    "canary_qwen_model_path": "/root/fawkes/models/canary-qwen-2.5b/",
    "ecapa_tdnn_model_path": "/root/fawkes/models/ecapa_tdnn_embed/ecapa_tdnn.nemo",
    "duckdb_path": "./speakers/speakers.duckdb",
    "server_name": "Fawkes",
    "default_speaker": "unknown speaker",
    "default_speaker_confidence": "uncertain",
    "default_asr_confidence": "certain",
    "rasa_url": "http://rasa-nlp:5005",  # Docker service name
    "rasa_timeout": 10,  # seconds
    "enable_rasa": True
}

# Initialize DuckDB connection at the top level
# This will create the database file if it doesn't exist
con = duckdb.connect(CONFIG["duckdb_path"])

def setup_database():
    """
    Sets up the DuckDB table for storing speaker data.
    This function should be called once at program startup.
    """
    con.execute("""
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
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("DuckDB table 'speakers' is ready.")

# Ensure the database connection is closed on exit
atexit.register(con.close)

active_websockets = {}  # client_id -> websocket
client_queues = {}
clientSideTTS = False

# Dialogue partners
#SPEAKER = "unknown speaker"
#SPEAKER_CONFIDENCE = "certain"
#SERVER = "Fawkes"
#ASR_CONFIDENCE = "certain"

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
        
        # Load the model directly from .safetensors format
        self.model = SALM.from_pretrained(str(self.model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Canary-Qwen-2.5b model loaded successfully")

    def transcribe_final(self, audio_int16, sample_rate=16000):
        """
        Perform final transcription with ITN and P&C using Canary-Qwen-2.5b
        Uses tensor input directly with model.generate() - no disk I/O
        
        Args:
            audio_int16 (np.ndarray): Audio data as int16 PCM (already 16kHz mono)
            sample_rate (int): Sample rate of the audio (should be 16000)
            
        Returns:
            str: Final transcription with punctuation and capitalization
        """
        try:
            print(f"[Canary-Qwen] Processing audio: shape={audio_int16.shape}, sample_rate={sample_rate}")
            
            # Convert int16 PCM to float32 tensor normalized to [-1, 1]
            # This is the standard format expected by speech models
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32).to(self.device)
            
            # Prepare batch format: (batch_size, time)
            audios = audio_tensor.unsqueeze(0)  # Add batch dimension
            audio_lens = torch.tensor([audios.shape[1]], dtype=torch.int64).to(self.device)
            
            # Prepare the prompt with audio locator tag
            prompts = [
                [{"role": "user", "content": f"Transcribe the following: {self.model.audio_locator_tag}"}]
            ]
            
            print("[Canary-Qwen] Running model.generate() with tensor input...")
            
            # Generate transcription using the model's generate method
            with torch.no_grad():
                raw_output_ids = self.model.generate(
                    prompts=prompts,
                    audios=audios,
                    audio_lens=audio_lens,
                    max_new_tokens=128,
                    do_sample=False,  # Deterministic output
                    num_beams=1,      # Greedy decoding for speed
                    temperature=1.0
                )
            
            # Decode the token IDs to text using the model's tokenizer
            raw_result = self.model.tokenizer.ids_to_text(raw_output_ids[0].cpu())
            print(f"[Canary-Qwen] Raw model output: '{raw_result}'")
            
            # Clean up the result - remove chat template artifacts
            result = raw_result.strip()
            
            # Handle chat template format if present
            if '<|im_start|>' in result:
                # Extract content between assistant tags
                parts = result.split('<|im_start|>assistant\n')
                if len(parts) > 1:
                    result = parts[1].split('<|im_end|>')[0].strip()
            
            # Remove any remaining template tokens
            result = result.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
            
            # Validate result quality
            if not result or result.lower() in ['transcript', 'transcription', 'audio transcript']:
                print(f"[Canary-Qwen] Got generic/empty response: '{result}'")
                return ""
            
            print(f"[Canary-Qwen] Final transcription: '{result}'")
            return result
            
        except Exception as e:
            print(f"[Canary-Qwen] Error in transcription: {e}")
            return ""

    def transcribe_with_beam_search(self, audio_int16, sample_rate=16000, num_beams=3):
        """
        Alternative method with beam search for potentially better quality
        
        Args:
            audio_int16 (np.ndarray): Audio data as int16 PCM (already 16kHz mono)
            sample_rate (int): Sample rate of the audio (should be 16000)
            num_beams (int): Number of beams for beam search decoding
            
        Returns:
            str: Final transcription with punctuation and capitalization
        """
        try:
            print(f"[Canary-Qwen] Running beam search transcription with {num_beams} beams")
            
            # Convert int16 PCM to float32 tensor normalized to [-1, 1]
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32).to(self.device)
            
            # Prepare batch format: (batch_size, time)
            audios = audio_tensor.unsqueeze(0)  # Add batch dimension
            audio_lens = torch.tensor([audios.shape[1]], dtype=torch.int64).to(self.device)
            
            # Prepare the prompt with audio locator tag
            prompts = [
                [{"role": "user", "content": f"Transcribe the following: {self.model.audio_locator_tag}"}]
            ]
            
            print(f"[Canary-Qwen] Running beam search with {num_beams} beams...")
            
            # Generate transcription using beam search
            with torch.no_grad():
                raw_output_ids = self.model.generate(
                    prompts=prompts,
                    audios=audios,
                    audio_lens=audio_lens,
                    max_new_tokens=128,
                    do_sample=False,      # Deterministic beam search
                    num_beams=num_beams,  # Use beam search
                    temperature=1.0
                )
            
            # Decode the token IDs to text
            raw_result = self.model.tokenizer.ids_to_text(raw_output_ids[0].cpu())
            print(f"[Canary-Qwen] Raw beam search output: '{raw_result}'")
            
            # Clean up the result
            result = raw_result.strip()
            
            # Handle chat template format if present
            if '<|im_start|>' in result:
                parts = result.split('<|im_start|>assistant\n')
                if len(parts) > 1:
                    result = parts[1].split('<|im_end|>')[0].strip()
            
            # Remove any remaining template tokens
            result = result.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
            
            # Validate result quality
            if not result or result.lower() in ['transcript', 'transcription', 'audio transcript']:
                print(f"[Canary-Qwen] Beam search got generic/empty response: '{result}'")
                return ""
            
            print(f"[Canary-Qwen] Beam search transcription: '{result}'")
            return result
            
        except Exception as e:
            print(f"[Canary-Qwen] Error in beam search transcription: {e}")
            return ""

class NeMoVAD:
    def __init__(self, model_path, device, sample_rate=16000):
        print("Pre-loading NeMo VAD model...")
        self.device = device
        self.sample_rate = sample_rate
        # The VAD model is an EncDecClassificationModel
        # Replace with EncDecSpeakerLabelModel older model deprecated
        self.model = nemo_asr.models.EncDecClassificationModel.restore_from(model_path, map_location=torch.device(self.device), strict=False)
        #self.model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name="vad_multilingual_marblenet", map_location=torch.device(self.device))
        #self.model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(model_name="nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0", map_location=torch.device(self.device))
        #nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name="vad_multilingual_marblenet")
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
        #print(f"PiperVoice methods: {[m for m in dir(self.voice) if not m.startswith('_')]}")

    # for Piper 1.2.0 (v1.3.0 represents major breaking changes)
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
        #self._load_speakers()
        self._load_speakers_from_db()

        print(f"Coqui XTTS Model loaded on device: {self.device}")
        #self._load_speakers()
        #print(f"Loaded {len(self.xtts_model.speaker_manager.speakers)} speakers.")

    @property
    def sample_rate(self) -> int:
        """Returns the sample rate of the XTTS model."""
        return self.config.audio["sample_rate"]

    async def extract_xtts_embed(self, wav_path, firstname, surname=None):
        """
        Extracts XTTS embeddings from a WAV file and stores them in the DuckDB database
        using native arrays and separate shape fields.
        
        Args:
            wav_path (str or Path): Path to the WAV file.
            firstname (str): The first name of the speaker.
            surname (str, optional): The last name of the speaker. Defaults to None.
        """
        wav_path = Path(wav_path)
        print(f"Extracting XTTS embeddings for {firstname} {surname if surname else ''} from {wav_path}...")
        
        with torch.no_grad():
            gpt_cond_latent, speaker_embedding = self.xtts_model.get_conditioning_latents(str(wav_path), 16000)
        
        print(f"Extracted shapes - GPT: {gpt_cond_latent.shape}, Speaker: {speaker_embedding.shape}")
        
        # Convert tensors to Python lists (flattened) for DuckDB array storage
        gpt_latent_flat = gpt_cond_latent.cpu().numpy().flatten().tolist()
        xtts_embedding_flat = speaker_embedding.cpu().numpy().flatten().tolist()
        
        # Convert shapes to JSON strings for storage
        #import json
        gpt_shape_json = json.dumps(list(gpt_cond_latent.shape))
        xtts_shape_json = json.dumps(list(speaker_embedding.shape))
        
        # Use asyncio's to_thread to run the blocking DB operation in a separate thread
        def insert_data():
            con.execute("""
                INSERT INTO speakers 
                (firstname, surname, gpt_cond_latent, gpt_shape, xtts_embedding, xtts_shape) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                firstname, 
                surname, 
                gpt_latent_flat,     # Native FLOAT[] array (flattened)
                gpt_shape_json,      # JSON string of original shape
                xtts_embedding_flat, # Native FLOAT[] array (flattened)
                xtts_shape_json      # JSON string of original shape
            ))
        
        await asyncio.to_thread(insert_data)
        
        print(f"Successfully stored XTTS embeddings for {firstname} in DuckDB using native arrays.")

    def _load_speakers_from_db(self):
        """
        Loads XTTS speaker embeddings from the DuckDB database using native arrays
        and separate shape fields.
        Private method called during initialization.
        """
        print("Loading XTTS speakers from DuckDB native arrays...")
        
        speakers_query = con.execute("""
            SELECT firstname, surname, gpt_cond_latent, gpt_shape, xtts_embedding, xtts_shape 
            FROM speakers 
            WHERE gpt_cond_latent IS NOT NULL AND xtts_embedding IS NOT NULL
        """).fetchall()
        
        # Reset the speaker manager dictionary
        # COMMENT OUT if you want to maintain preloaded voices from XTTS
        self.xtts_model.speaker_manager.speakers = {}
        
        for row in speakers_query:
            firstname, surname, gpt_latent_list, gpt_shape_json, xtts_emb_list, xtts_shape_json = row
            
            # Reconstruct the speaker name
            speaker_name = f"{firstname}_{surname}" if surname else firstname
            
            try:
                import json
                
                # Parse the shape information from JSON
                gpt_shape = tuple(json.loads(gpt_shape_json))
                xtts_shape = tuple(json.loads(xtts_shape_json))
                
                # Convert from DuckDB arrays (Python lists) back to numpy arrays with proper shapes
                gpt_latent = torch.from_numpy(
                    np.array(gpt_latent_list, dtype=np.float32).reshape(gpt_shape)
                ).to(self.device)
                
                xtts_embedding = torch.from_numpy(
                    np.array(xtts_emb_list, dtype=np.float32).reshape(xtts_shape)
                ).to(self.device)
                
                # will eventually modify this function to load speakers with id instead of name
                self.xtts_model.speaker_manager.speakers[speaker_name] = {
                    "gpt_cond_latent": gpt_latent,
                    "speaker_embedding": xtts_embedding
                }
                
                print(f"Loaded {speaker_name} with shapes - GPT: {gpt_latent.shape}, Speaker: {xtts_embedding.shape}")
                
            except Exception as e:
                print(f"Error loading speaker {speaker_name}: {e}")
                continue
        
        print(f"Loaded {len(self.xtts_model.speaker_manager.speakers)} XTTS speakers from DuckDB native arrays.")

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
            language="en", # Assuming English for now
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            stream_chunk_size=512, # You can adjust this for latency vs. throughput
        ):
            # Convert PyTorch tensor to NumPy array on CPU before yielding
            # (assuming subsequent processing expects NumPy)
            yield chunk.cpu().numpy()

class FastECAPASpeakerMatcher:
    """
    In-memory speaker matcher for ECAPA-TDNN embeddings with adaptive confidence scoring.
    Loads all speaker embeddings once at startup for fast comparisons.
    """
    
    def __init__(self, duckdb_connection):
        """
        Initialize the matcher and load all ECAPA embeddings into memory.
        
        Args:
            duckdb_connection: Active DuckDB connection object
        """
        self.con = duckdb_connection
        self.speaker_embeddings = {}  # speaker_name -> numpy array
        self.embedding_matrix = None  # Combined matrix for vectorized operations
        self.speaker_names = []       # Ordered list matching matrix rows
        self.speaker_uids = []        # Ordered list of UIDs matching matrix rows
        self.load_embeddings_to_memory()
    
    def load_embeddings_to_memory(self):
        """
        Load all ECAPA embeddings from database into memory for fast comparison.
        Called once during initialization.
        """
        print("Loading ECAPA embeddings into memory...")
        
        try:
            # Query all speakers with ECAPA embeddings
            speakers_query = self.con.execute("""
                SELECT uid, firstname, surname, ecapa_embedding 
                FROM speakers 
                WHERE ecapa_embedding IS NOT NULL
            """).fetchall()
            
            if not speakers_query:
                print("No ECAPA embeddings found in database.")
                return
            
            embeddings_list = []
            
            for row in speakers_query:
                uid, firstname, surname, ecapa_embedding_list = row
                
                # Reconstruct speaker name
                speaker_name = f"{firstname}_{surname}" if surname else firstname
                
                try:
                    # Convert DuckDB array (Python list) to numpy array
                    embedding_array = np.array(ecapa_embedding_list, dtype=np.float32)
                    
                    # Normalize the embedding for cosine similarity
                    embedding_normalized = embedding_array / np.linalg.norm(embedding_array)
                    
                    # Store in dictionary
                    self.speaker_embeddings[speaker_name] = embedding_normalized
                    
                    # Add to list for matrix construction
                    embeddings_list.append(embedding_normalized)
                    self.speaker_names.append(speaker_name)
                    self.speaker_uids.append(uid)
                    
                    print(f"Loaded ECAPA embedding for {speaker_name} (shape: {embedding_array.shape})")
                    
                except Exception as e:
                    print(f"Error loading ECAPA embedding for {speaker_name}: {e}")
                    continue
            
            if embeddings_list:
                # Create combined matrix for vectorized operations
                self.embedding_matrix = np.vstack(embeddings_list)
                print(f"Created embedding matrix: {self.embedding_matrix.shape}")
                print(f"Loaded {len(self.speaker_embeddings)} ECAPA embeddings into memory.")
            else:
                print("No valid ECAPA embeddings loaded.")
                
        except Exception as e:
            print(f"Error loading ECAPA embeddings: {e}")
    
    def calculate_adaptive_confidence(self, best_score: float, second_score: float, domain_size: int) -> float:
        """
        Calculate confidence score that adapts based on domain size.
        
        Args:
            best_score: Highest cosine similarity score
            second_score: Second highest cosine similarity score
            domain_size: Number of speakers in consideration domain
            
        Returns:
            Composite confidence score (0.0 to 1.0)
        """
        base_confidence = best_score
        gap = best_score - second_score
        
        # Dynamic gap weighting: inversely proportional to domain size
        # Tunable parameters:
        # - 12.0: Controls gap sensitivity (higher = more sensitive for small domains)
        # - 0.4: Maximum gap weight ceiling
        gap_weight = min(0.4, 12.0 / domain_size)
        
        gap_bonus = gap * gap_weight
        composite = min(1.0, base_confidence + gap_bonus)
        
        return composite
    
    def find_best_match_with_nomatch_data(self, query_embedding, domain_size: Optional[int] = None) -> Tuple[Optional[str], Optional[int], float, Dict]:
        """
        Find the best matching speaker with additional data needed for nomatch scoring.
        
        Args:
            query_embedding: ECAPA embedding to match (torch.Tensor or numpy array)
            domain_size: Number of speakers in consideration domain 
                        (defaults to total database size if not specified)
            
        Returns:
            Tuple of (speaker_name, uid, confidence_score, nomatch_data_dict)
        """
        if self.embedding_matrix is None or len(self.speaker_embeddings) == 0:
            return None, None, 0.0, {"error": "No embeddings loaded"}
        
        # Use total database size if domain_size not specified
        if domain_size is None:
            domain_size = len(self.speaker_embeddings)
        
        try:
            # Handle both torch.Tensor and numpy array inputs
            if isinstance(query_embedding, torch.Tensor):
                query_array = query_embedding.detach().cpu().numpy()
            else:
                query_array = query_embedding
            
            # Normalize the query embedding
            query_normalized = query_array / np.linalg.norm(query_array)
            query_normalized = query_normalized.reshape(1, -1)
            
            # Compute cosine similarities with all speakers at once (vectorized)
            similarities = cosine_similarity(query_normalized, self.embedding_matrix)[0]
            
            # Get top scores for confidence calculation
            sorted_indices = np.argsort(similarities)[::-1]  # Sort descending
            best_score = similarities[sorted_indices[0]]
            second_score = similarities[sorted_indices[1]] if len(similarities) > 1 else 0.0
            
            # Additional statistics for nomatch calculation
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            median_similarity = np.median(similarities)
            
            # Get best matching speaker and UID
            best_speaker = self.speaker_names[sorted_indices[0]]
            best_uid = self.speaker_uids[sorted_indices[0]]
            
            # Calculate adaptive confidence
            confidence = self.calculate_adaptive_confidence(best_score, second_score, domain_size)
            
            # Prepare nomatch data
            nomatch_data = {
                "best_similarity": best_score,
                "second_similarity": second_score,
                "mean_similarity": mean_similarity,
                "std_similarity": std_similarity,
                "median_similarity": median_similarity,
                "similarity_gap": best_score - second_score,
                "domain_size": domain_size,
                "above_median_count": np.sum(similarities > median_similarity),
                "cosine_dissimilarity": 1.0 - best_score  # Direct dissimilarity measure
            }
            
            return best_speaker, best_uid, confidence, nomatch_data
            
        except Exception as e:
            print(f"Error in speaker matching: {e}")
            return None, None, 0.0, {"error": str(e)}
    
    def find_best_match(self, query_embedding, domain_size: Optional[int] = None) -> Tuple[Optional[str], Optional[int], float]:
        """
        Find the best matching speaker with adaptive confidence scoring.
        
        Args:
            query_embedding: ECAPA embedding to match (torch.Tensor or numpy array)
            domain_size: Number of speakers in consideration domain 
                        (defaults to total database size if not specified)
            
        Returns:
            Tuple of (speaker_name, uid, confidence_score) or (None, None, 0.0) if no embeddings loaded
        """
        if self.embedding_matrix is None or len(self.speaker_embeddings) == 0:
            return None, None, 0.0
        
        # Use total database size if domain_size not specified
        if domain_size is None:
            domain_size = len(self.speaker_embeddings)
        
        try:
            # Handle both torch.Tensor and numpy array inputs
            if isinstance(query_embedding, torch.Tensor):
                query_array = query_embedding.detach().cpu().numpy()
            else:
                query_array = query_embedding
            
            # Normalize the query embedding
            query_normalized = query_array / np.linalg.norm(query_array)
            query_normalized = query_normalized.reshape(1, -1)
            
            # Compute cosine similarities with all speakers at once (vectorized)
            similarities = cosine_similarity(query_normalized, self.embedding_matrix)[0]
            
            # Get top 2 scores for confidence calculation
            sorted_indices = np.argsort(similarities)[::-1]  # Sort descending
            best_score = similarities[sorted_indices[0]]
            second_score = similarities[sorted_indices[1]] if len(similarities) > 1 else 0.0
            
            # Get best matching speaker and UID
            best_speaker = self.speaker_names[sorted_indices[0]]
            best_uid = self.speaker_uids[sorted_indices[0]]
            
            # Calculate adaptive confidence
            confidence = self.calculate_adaptive_confidence(best_score, second_score, domain_size)
            
            return best_speaker, best_uid, confidence
            
        except Exception as e:
            print(f"Error in speaker matching: {e}")
            return None, None, 0.0
    
    def find_best_match_with_details(self, query_embedding, domain_size: Optional[int] = None) -> Dict:
        """
        Find best match with detailed breakdown for debugging/analysis.
        
        Args:
            query_embedding: ECAPA embedding to match (torch.Tensor or numpy array)
            domain_size: Number of speakers in consideration domain
        
        Returns:
            Dictionary with detailed matching information
        """
        if self.embedding_matrix is None or len(self.speaker_embeddings) == 0:
            return {"speaker": None, "confidence": 0.0, "details": "No embeddings loaded"}
        
        if domain_size is None:
            domain_size = len(self.speaker_embeddings)
        
        try:
            # Handle both torch.Tensor and numpy array inputs
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
        """Get the total number of speakers loaded in memory."""
        return len(self.speaker_embeddings)
    
    def rebuild_embedding_matrix(self):
        """Rebuild embedding matrix from database (useful after new speaker enrollment)."""
        print("Rebuilding ECAPA embedding matrix from database...")
        self.speaker_embeddings.clear()
        self.speaker_names.clear()
        self.speaker_uids.clear()
        self.embedding_matrix = None
        self.load_embeddings_to_memory()

    def update_embedding_in_matrix(self, uid: int, speaker_name: str, new_embedding: np.ndarray) -> bool:
        """
        Update a specific speaker's embedding in the matrix without full rebuild.
        
        Args:
            uid: Speaker's UID in database (used for lookup)
            speaker_name: Speaker name for logging purposes only
            new_embedding: New ECAPA embedding as numpy array
            
        Returns:
            bool: True if successful, False if speaker not found
        """
        try:
            # Check if we have embeddings loaded
            if self.embedding_matrix is None or len(self.speaker_embeddings) == 0:
                print(f"[ECAPA Matcher] No embeddings loaded for {speaker_name}, performing full rebuild...")
                self.rebuild_embedding_matrix()
                return True
            
            # Find speaker by UID
            try:
                speaker_idx = self.speaker_uids.index(uid)
                actual_speaker_name = self.speaker_names[speaker_idx]
            except ValueError:
                print(f"[ECAPA Matcher] Speaker {speaker_name} (UID: {uid}) not found in matrix, adding new speaker...")
                # There should really be no reason this could ever happen
                return self.add_new_speaker_to_matrix(uid, speaker_name, new_embedding)
            
            # Normalize the new embedding
            embedding_normalized = new_embedding / np.linalg.norm(new_embedding)
            
            # Update the dictionary (use actual speaker name from matrix)
            self.speaker_embeddings[actual_speaker_name] = embedding_normalized
            
            # Update the matrix in-place
            self.embedding_matrix[speaker_idx] = embedding_normalized
            
            print(f"[ECAPA Matcher] Updated embedding for {speaker_name} (UID: {uid}) in-place")
            return True
            
        except Exception as e:
            print(f"[ECAPA Matcher] Error updating speaker embedding for {speaker_name} (UID: {uid}): {e}")
            return False

    def add_new_speaker_to_matrix(self, uid: int, speaker_name: str, new_embedding: np.ndarray) -> bool:
        """
        Add a completely new speaker to the matrix.
        
        Args:
            uid: Speaker's UID in database
            speaker_name: Speaker name
            new_embedding: ECAPA embedding as numpy array
            
        Returns:
            bool: True if successful
        """
        try:
            # Check if UID already exists (shouldn't happen, but safety check)
            if uid in self.speaker_uids:
                existing_idx = self.speaker_uids.index(uid)
                existing_name = self.speaker_names[existing_idx]
                print(f"[ECAPA Matcher] Warning: UID {uid} already exists for {existing_name}, updating instead...")
                return self.update_embedding_in_matrix(uid, speaker_name, new_embedding)
            
            # Normalize the embedding
            embedding_normalized = new_embedding / np.linalg.norm(new_embedding)
            
            # Add to dictionary
            self.speaker_embeddings[speaker_name] = embedding_normalized
            
            # Add to lists
            self.speaker_names.append(speaker_name)
            self.speaker_uids.append(uid)
            
            # Expand the matrix
            if self.embedding_matrix is None:
                # First speaker
                self.embedding_matrix = embedding_normalized.reshape(1, -1)
            else:
                # Add new row to existing matrix
                new_row = embedding_normalized.reshape(1, -1)
                self.embedding_matrix = np.vstack([self.embedding_matrix, new_row])
            
            print(f"[ECAPA Matcher] Added new speaker {speaker_name} (UID: {uid}) to matrix (new size: {self.embedding_matrix.shape})")
            return True
            
        except Exception as e:
            print(f"[ECAPA Matcher] Error adding new speaker {speaker_name} (UID: {uid}): {e}")
            return False

    def remove_speaker_from_matrix(self, uid: int, speaker_name: str = None) -> bool:
        """
        Remove a speaker from the matrix by UID.
        
        Args:
            uid: Speaker's UID to remove
            speaker_name: Optional speaker name for logging purposes
            
        Returns:
            bool: True if successful
        """
        try:
            # Find speaker by UID
            try:
                speaker_idx = self.speaker_uids.index(uid)
                actual_speaker_name = self.speaker_names[speaker_idx]
            except ValueError:
                log_name = speaker_name if speaker_name else f"UID {uid}"
                print(f"[ECAPA Matcher] Speaker {log_name} not found for removal")
                return False
            
            # Remove from dictionary
            del self.speaker_embeddings[actual_speaker_name]
            
            # Remove from lists
            self.speaker_names.pop(speaker_idx)
            self.speaker_uids.pop(speaker_idx)
            
            # Remove row from matrix
            if self.embedding_matrix.shape[0] == 1:
                # Last speaker
                self.embedding_matrix = None
            else:
                # Remove specific row
                self.embedding_matrix = np.delete(self.embedding_matrix, speaker_idx, axis=0)
            
            log_name = speaker_name if speaker_name else actual_speaker_name
            print(f"[ECAPA Matcher] Removed speaker {log_name} (UID: {uid}) from matrix")
            return True
            
        except Exception as e:
            log_name = speaker_name if speaker_name else f"UID {uid}"
            print(f"[ECAPA Matcher] Error removing speaker {log_name}: {e}")
            return False

class ECAPASpeakerProcessor:
    """
    Unified ECAPA-TDNN speaker processor that handles both file-based extraction 
    for initial speaker imprints and live audio buffer processing for real-time identification.
    """
    
    def __init__(self, model_path, device, ecapa_matcher, sample_rate=16000):
        """
        Initialize the ECAPA-TDNN model and processor.
        
        Args:
            model_path (str): Path to the .nemo model file
            device (str): Device to load the model on ('cuda' or 'cpu')
            ecapa_matcher (FastECAPASpeakerMatcher): The speaker matching system
            sample_rate (int): Audio sample rate (default: 16000)
        """
        print("Pre-loading ECAPA-TDNN speaker processor...")
        self.device = device
        self.model_path = model_path
        self.ecapa_matcher = ecapa_matcher
        self.sample_rate = sample_rate
        
        # Load the ECAPA-TDNN model
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
            model_path, 
            map_location=torch.device(self.device)
        )
        self.model.eval()
        self.model.to(self.device)
        
        # Online processing state
        self.bytes_per_second = sample_rate * 2  # 16-bit audio = 2 bytes per sample
        self.last_extraction_bytes = 0
        self.extraction_interval_bytes = self.bytes_per_second  # Extract every 1 second
        self.max_extractions = 7  # Stop after 7 seconds
        self.extraction_count = 0
        self.subsequent_nomatch = 0
        self.total_nomatch = 0
        # Thresholds for nomatch determination
        self.nomatch_lower_threshold = 0.70
        self.nomatch_upper_threshold = 0.85
        # Thresholds for speaker identification
        self.UNCERTAIN_THRESHOLD = 0.70
        self.CERTAIN_THRESHOLD = 0.85
        
        print("ECAPA-TDNN speaker processor loaded successfully")
    
    def extract_embedding_from_file(self, wav_path, sample_rate=None):
        """
        Extract ECAPA embedding from a WAV file using the model's built-in file handling.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        try:
            print(f"[ECAPA] Extracting embedding from file: {wav_path}")
            
            # The model's get_embedding method is designed to handle file paths.
            with torch.no_grad():
                embeddings = self.model.get_embedding(wav_path)
            
            embedding_np = embeddings.cpu().numpy().squeeze()
            
            print(f"[ECAPA] Extracted embedding shape: {embedding_np.shape}")
            return embedding_np
        except Exception as e:
            print(f"[ECAPA] Error extracting embedding from file {wav_path}: {e}")
            return None

    def extract_embedding_from_buffer(self, audio_int16, sample_rate=None):
        """
        Extract ECAPA embedding from int16 PCM audio buffer using the model's forward method.
        Fixed to follow the working patterns from claudeECAPAfromTensor2.py
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        try:
            #print(f"[ECAPA] Extracting embedding from buffer: shape={audio_int16.shape}, sample_rate={sample_rate}")
            
            # Convert int16 to float32 in range [-1, 1] (same as working version)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Convert to torch tensor
            waveform = torch.from_numpy(audio_float32)
            
            # Ensure single channel (mono) - follow claudeECAPAfromTensor2.py pattern
            if waveform.shape[0] > 1 if waveform.dim() > 1 else False:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif waveform.dim() == 1:
                # Add channel dimension if it's missing: [T] -> [1, T]
                waveform = waveform.unsqueeze(0)
            
            # Resample if necessary (ECAPA-TDNN typically expects 16kHz)
            target_sr = 16000
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            #print(f"[ECAPA] Prepared waveform tensor shape: {waveform.shape}")
            #print(f"[ECAPA] Sample rate: {target_sr}")
            
            # Move tensors to the same device as the model
            device = next(self.model.parameters()).device
            waveform = waveform.to(device)
            
            # Generate embedding
            with torch.no_grad():
                # NeMo models typically expect audio length in samples as second parameter
                audio_length = torch.tensor([waveform.shape[1]], dtype=torch.long).to(device)
                
                # Use the EXACT same forward call pattern as the working version
                _, embedding = self.model.forward(waveform, audio_length)
            
            embedding_np = embedding.cpu().numpy().squeeze()
            
            #print(f"[ECAPA] Extracted embedding shape from buffer: {embedding_np.shape}")
            return embedding_np
            
        except Exception as e:
            print(f"[ECAPA] Error extracting embedding from buffer: {e}")
            return None

    async def create_initial_speaker_imprint(self, wav_path, firstname, surname=None):
        """
        Extract both XTTS and ECAPA embeddings from a WAV file and store them in the database.
        This replaces the old extract_xtts_embed function with expanded functionality.
        
        Args:
            wav_path (str or Path): Path to the WAV file
            firstname (str): The first name of the speaker
            surname (str, optional): The last name of the speaker. Defaults to None.
            
        Returns:
            bool: True if successful, False otherwise
        """
        global xtts_wrapper, con
        
        wav_path = Path(wav_path)
        print(f"[ECAPA] Creating initial speaker imprint for {firstname} {surname if surname else ''} from {wav_path}...")
        
        try:
            # Get duration of audio file
            try:
                #import librosa
                audio_duration = librosa.get_duration(path=str(wav_path))
                print(f"[ECAPA] Audio duration: {audio_duration:.2f} seconds")
            except Exception as e:
                print(f"[ECAPA] Error getting audio duration: {e}")
                audio_duration = 0.0
            
            # Extract XTTS embeddings
            print(f"[ECAPA] Extracting XTTS embeddings...")
            with torch.no_grad():
                gpt_cond_latent, speaker_embedding = xtts_wrapper.xtts_model.get_conditioning_latents(str(wav_path), 16000)
            
            print(f"[ECAPA] XTTS shapes - GPT: {gpt_cond_latent.shape}, Speaker: {speaker_embedding.shape}")
            
            # Convert XTTS tensors to Python lists (flattened) for DuckDB array storage
            gpt_latent_flat = gpt_cond_latent.cpu().numpy().flatten().tolist()
            xtts_embedding_flat = speaker_embedding.cpu().numpy().flatten().tolist()
            
            # Convert XTTS shapes to JSON strings for storage
            gpt_shape_json = json.dumps(list(gpt_cond_latent.shape))
            xtts_shape_json = json.dumps(list(speaker_embedding.shape))
            
            # Extract ECAPA embedding
            print(f"[ECAPA] Extracting ECAPA embedding...")
            ecapa_embedding = await asyncio.to_thread(
                self.extract_embedding_from_file, 
                wav_path, 
                self.sample_rate
            )
            
            if ecapa_embedding is None:
                print(f"[ECAPA] Failed to extract ECAPA embedding, storing XTTS data only")
                ecapa_embedding_flat = None
            else:
                # Convert ECAPA embedding to Python list for DuckDB array storage
                ecapa_embedding_flat = ecapa_embedding.flatten().tolist()
                print(f"[ECAPA] ECAPA embedding shape: {ecapa_embedding.shape}")
            
            # Store all embeddings in database using asyncio thread pool
            def insert_speaker_data():
                con.execute("""
                    INSERT INTO speakers 
                    (firstname, surname, gpt_cond_latent, gpt_shape, xtts_embedding, xtts_shape, 
                    ecapa_embedding, total_duration_sec, sample_count, last_updated) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    firstname, 
                    surname, 
                    gpt_latent_flat,     # XTTS GPT latent (flattened)
                    gpt_shape_json,      # XTTS GPT shape as JSON
                    xtts_embedding_flat, # XTTS speaker embedding (flattened)
                    xtts_shape_json,     # XTTS speaker shape as JSON
                    ecapa_embedding_flat,# ECAPA embedding (flattened)
                    audio_duration,      # Initial duration
                    1,                   # Initial sample count
                ))
            
            await asyncio.to_thread(insert_speaker_data)
            
            print(f"[ECAPA] Successfully stored complete speaker imprint for {firstname} in DuckDB")
            print(f"[ECAPA] Initial metadata - Duration: {audio_duration:.2f}s, Sample count: 1")
            
            # Reload the XTTS speaker manager to include the new speaker
            xtts_wrapper._load_speakers_from_db()
            
            # Rebuild the ECAPA matcher embedding matrix to include the new speaker
            self.ecapa_matcher.rebuild_embedding_matrix()
            
            return True
            
        except Exception as e:
            print(f"[ECAPA] Error creating speaker imprint for {firstname}: {e}")
            return False

    async def update_speaker_imprint_from_file(self, wav_path, uid):
        """
        Perform a cumulative update to an existing speaker's ECAPA embedding.
        Uses weighted averaging based on audio duration to combine old and new embeddings.
        
        Args:
            wav_path (str or Path): Path to the new WAV file
            uid (int): The UID of the existing speaker
            
        Returns:
            bool: True if successful, False otherwise
        """
        wav_path = Path(wav_path)
        print(f"[ECAPA] Updating speaker imprint for UID {uid} with {wav_path}...")
        
        try:
            # 1. Check if the speaker exists
            def check_speaker_exists():
                result = con.execute("""
                    SELECT uid, firstname, surname, ecapa_embedding, total_duration_sec, sample_count 
                    FROM speakers 
                    WHERE uid = ?
                """, (uid,)).fetchone()
                return result
            
            existing_speaker = await asyncio.to_thread(check_speaker_exists)
            
            if existing_speaker is None:
                print(f"[ECAPA] Error: Speaker with UID {uid} does not exist in database")
                return False
            
            uid_db, firstname, surname, existing_embedding_list, total_duration, sample_count = existing_speaker
            speaker_name = f"{firstname}_{surname}" if surname else firstname
            print(f"[ECAPA] Found existing speaker: {speaker_name}")
            
            # 2. Get duration of new audio file
            try:
                #import librosa
                new_audio_duration = librosa.get_duration(path=str(wav_path))
                print(f"[ECAPA] New audio duration: {new_audio_duration:.2f} seconds")
            except Exception as e:
                print(f"[ECAPA] Error getting audio duration: {e}")
                return False
            
            # 3. Extract new ECAPA embedding from the WAV file
            new_embedding = await asyncio.to_thread(
                self.extract_embedding_from_file, 
                wav_path, 
                self.sample_rate
            )
            
            if new_embedding is None:
                print(f"[ECAPA] Failed to extract ECAPA embedding from {wav_path}")
                return False
            
            print(f"[ECAPA] Successfully extracted new embedding shape: {new_embedding.shape}")
            
            # 4. Combine embeddings using weighted average
            if existing_embedding_list is not None and total_duration > 0:
                # Convert existing embedding from database list back to numpy array
                existing_embedding = np.array(existing_embedding_list, dtype=np.float32)
                
                # Calculate weights based on duration
                existing_weight = total_duration / (total_duration + new_audio_duration)
                new_weight = new_audio_duration / (total_duration + new_audio_duration)
                
                # Perform weighted average
                combined_embedding = (existing_weight * existing_embedding + 
                                    new_weight * new_embedding)
                
                print(f"[ECAPA] Combined embeddings - existing weight: {existing_weight:.3f}, "
                    f"new weight: {new_weight:.3f}")
                
            else:
                # No existing embedding or duration, use the new embedding as-is
                combined_embedding = new_embedding
                print(f"[ECAPA] No existing embedding data, using new embedding as baseline")
            
            # 5. Update database with combined embedding and metadata
            def update_speaker_data():
                combined_embedding_list = combined_embedding.flatten().tolist()
                new_total_duration = (total_duration if total_duration else 0.0) + new_audio_duration
                new_sample_count = (sample_count if sample_count else 0) + 1
                
                con.execute("""
                    UPDATE speakers 
                    SET ecapa_embedding = ?, 
                        total_duration_sec = ?, 
                        sample_count = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE uid = ?
                """, (
                    combined_embedding_list,
                    new_total_duration,
                    new_sample_count,
                    uid
                ))
                
                return new_total_duration, new_sample_count
            
            new_total_duration, new_sample_count = await asyncio.to_thread(update_speaker_data)
            
            print(f"[ECAPA] Successfully updated speaker {speaker_name} (UID: {uid})")
            print(f"[ECAPA] New totals - Duration: {new_total_duration:.2f}s, Samples: {new_sample_count}")
            
            # 6. Rebuild the ECAPA matcher embedding matrix to include updated embedding
            print(f"[ECAPA] Rebuilding embedding matrix with updated data...")
            self.ecapa_matcher.rebuild_embedding_matrix()
            
            return True
            
        except Exception as e:
            print(f"[ECAPA] Error updating speaker imprint for UID {uid}: {e}")
            return False

    async def update_speaker_imprint_from_buffer(self, uid, ecapa_embedding, audio_int16, sample_rate=None):
        """
        Perform a cumulative update to an existing speaker's ECAPA embedding using audio buffer data.
        Uses weighted averaging based on audio duration to combine old and new embeddings.
        
        Args:
            uid (int): The UID of the existing speaker
            ecapa_embedding (np.ndarray): Pre-computed ECAPA embedding
            audio_int16 (np.ndarray): Audio data as int16 PCM array
            sample_rate (int, optional): Sample rate of the audio. Defaults to self.sample_rate.
            
        Returns:
            bool: True if successful, False otherwise
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        print(f"[ECAPA] Updating speaker imprint for UID {uid} from audio buffer...")
        
        try:
            # 1. Check if the speaker exists
            def check_speaker_exists():
                result = con.execute("""
                    SELECT uid, firstname, surname, ecapa_embedding, total_duration_sec, sample_count 
                    FROM speakers 
                    WHERE uid = ?
                """, (uid,)).fetchone()
                return result
            
            existing_speaker = await asyncio.to_thread(check_speaker_exists)
            
            if existing_speaker is None:
                print(f"[ECAPA] Error: Speaker with UID {uid} does not exist in database")
                return False
            
            uid_db, firstname, surname, existing_embedding_list, total_duration, sample_count = existing_speaker
            speaker_name = f"{firstname}_{surname}" if surname else firstname
            print(f"[ECAPA] Found existing speaker: {speaker_name}")
            
            # 2. Calculate duration from audio buffer
            try:
                # Calculate duration: number of samples / sample rate
                audio_duration_samples = len(audio_int16)
                new_audio_duration = audio_duration_samples / sample_rate
                print(f"[ECAPA] Audio buffer duration: {new_audio_duration:.2f} seconds ({audio_duration_samples} samples at {sample_rate}Hz)")
            except Exception as e:
                print(f"[ECAPA] Error calculating audio duration: {e}")
                return False
            
            # 3. Validate the pre-computed embedding
            if ecapa_embedding is None:
                print(f"[ECAPA] Error: Pre-computed ECAPA embedding is None")
                return False
                
            if not isinstance(ecapa_embedding, np.ndarray):
                print(f"[ECAPA] Error: ECAPA embedding must be a numpy array, got {type(ecapa_embedding)}")
                return False
                
            print(f"[ECAPA] Using pre-computed embedding shape: {ecapa_embedding.shape}")
            
            # 4. Combine embeddings using weighted average
            if existing_embedding_list is not None and total_duration > 0:
                # Convert existing embedding from database list back to numpy array
                existing_embedding = np.array(existing_embedding_list, dtype=np.float32)
                
                # Calculate weights based on duration
                existing_weight = total_duration / (total_duration + new_audio_duration)
                new_weight = new_audio_duration / (total_duration + new_audio_duration)
                
                # Perform weighted average
                combined_embedding = (existing_weight * existing_embedding + 
                                    new_weight * ecapa_embedding)
                
                print(f"[ECAPA] Combined embeddings - existing weight: {existing_weight:.3f}, "
                    f"new weight: {new_weight:.3f}")
                
            else:
                # No existing embedding or duration, use the new embedding as-is
                combined_embedding = ecapa_embedding
                print(f"[ECAPA] No existing embedding data, using new embedding as baseline")
            
            # 5. Update database with combined embedding and metadata
            def update_speaker_data():
                combined_embedding_list = combined_embedding.flatten().tolist()
                new_total_duration = (total_duration if total_duration else 0.0) + new_audio_duration
                new_sample_count = (sample_count if sample_count else 0) + 1
                
                con.execute("""
                    UPDATE speakers 
                    SET ecapa_embedding = ?, 
                        total_duration_sec = ?, 
                        sample_count = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE uid = ?
                """, (
                    combined_embedding_list,
                    new_total_duration,
                    new_sample_count,
                    uid
                ))
                
                return new_total_duration, new_sample_count
            
            new_total_duration, new_sample_count = await asyncio.to_thread(update_speaker_data)
            
            print(f"[ECAPA] Successfully updated speaker {speaker_name} (UID: {uid}) from buffer")
            print(f"[ECAPA] New totals - Duration: {new_total_duration:.2f}s, Samples: {new_sample_count}")
            
            # 6. Update the ECAPA matcher embedding matrix incrementally (much faster than rebuild)
            update_success = self.ecapa_matcher.update_embedding_in_matrix(uid, speaker_name, combined_embedding)
            if not update_success:
                print(f"[ECAPA] Warning: Failed to update embedding matrix, falling back to full rebuild...")
                self.ecapa_matcher.rebuild_embedding_matrix()
            
            return True
            
        except Exception as e:
            print(f"[ECAPA] Error updating speaker imprint for UID {uid} from buffer: {e}")
            return False

    def reset_for_new_utterance(self):
        """Reset the online processor state for a new utterance."""
        self.last_extraction_bytes = 0
        self.extraction_count = 0
        #self.subsequent_nomatch = 0
        #print("[ECAPA] Reset for new utterance")
    
    def should_extract_now(self, buffer_size_bytes):
        """
        Determine if we should extract an embedding based on buffer size.
        
        Args:
            buffer_size_bytes (int): Current size of the audio buffer in bytes
            
        Returns:
            bool: True if we should extract an embedding now
        """
        if self.extraction_count >= self.max_extractions:
            return False
        
        bytes_since_last = buffer_size_bytes - self.last_extraction_bytes
        return bytes_since_last >= self.extraction_interval_bytes
    
    def calculate_nomatch_score(self, nomatch_data: Dict, buffer_duration: float) -> float:
        """
        Calculate the probability that the speaker is NOT in the database.
        
        Args:
            nomatch_data: Dictionary containing similarity statistics from matching
            buffer_duration: Duration of the audio buffer in seconds
            
        Returns:
            float: Nomatch score (0.0 to 1.0), where higher means more likely to be unknown speaker
        """
        if "error" in nomatch_data:
            return 0.5  # Neutral score if we can't calculate
        
        # Base dissimilarity score (primary factor)
        base_dissimilarity = nomatch_data["cosine_dissimilarity"]
        
        # Duration confidence factor
        # Research suggests 2-3 seconds is typically sufficient for reliable speaker embeddings
        # Very short utterances are unreliable, but diminishing returns after ~3 seconds
        min_duration = 0.8  # Below this, unreliable (less than ~13 phonemes)
        optimal_duration = 3.0  # Above this, marginal improvement
        
        if buffer_duration <= min_duration:
            # Penalize very short utterances heavily
            duration_reliability = max(0.05, buffer_duration / min_duration * 0.3)
        elif buffer_duration >= optimal_duration:
            duration_reliability = 1.0  # Full confidence at 3+ seconds
        else:
            # Steep curve from 0.3 to 1.0 between min_duration and optimal_duration
            progress = (buffer_duration - min_duration) / (optimal_duration - min_duration)
            duration_reliability = 0.3 + 0.7 * (progress ** 0.7)  # Slightly accelerated curve
        
        # Domain size factor (larger databases need higher dissimilarity to be confident about nomatch)
        # Use smooth scaling similar to calculate_adaptive_confidence()
        domain_size = nomatch_data["domain_size"]
        # Tunable parameters:
        # - 15.0: Controls sensitivity to domain size (higher = more sensitive)
        # - 0.7: Minimum adjustment factor for very large domains
        # - 0.3: Maximum reduction from baseline (1.0 - 0.7 = 0.3)
        domain_adjustment = max(0.7, 1.0 - (0.3 * domain_size / (domain_size + 15.0)))
        
        # Statistical outlier factor
        # If the best match is far below the mean, it's more likely to be unknown
        mean_sim = nomatch_data["mean_similarity"]
        best_sim = nomatch_data["best_similarity"]
        std_sim = nomatch_data["std_similarity"]
        
        if std_sim > 0:
            z_score = (best_sim - mean_sim) / std_sim
            # Negative z-scores (below mean) increase nomatch likelihood
            outlier_factor = max(0.0, -z_score * 0.1)  # Cap the bonus
            outlier_factor = min(0.3, outlier_factor)   # Limit maximum impact
        else:
            outlier_factor = 0.0
        
        # Combine factors
        raw_nomatch_score = (base_dissimilarity * duration_reliability * domain_adjustment) + outlier_factor
        
        # Final normalization and bounds checking
        nomatch_score = min(1.0, max(0.0, raw_nomatch_score))
        
        return nomatch_score

    async def extract_and_match_from_buffer(self, audio_buffer, reason="scheduled"):
        """
        Extract ECAPA embedding from audio buffer and find best speaker match.
        
        Args:
            audio_buffer (bytes): Audio buffer as int16 PCM bytes
            reason (str): Reason for extraction ("scheduled", "silence", etc.)
            
        Returns:
            dict: Results containing speaker match information
        """
        try:
            buffer_duration = len(audio_buffer) / self.bytes_per_second
            #print(f"[ECAPA] Extracting embedding ({reason}) - buffer duration: {buffer_duration:.2f}s")
            
            # Convert bytes to numpy array
            audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16)
            
            # Extract ECAPA embedding in thread pool to avoid blocking
            ecapa_embedding = await asyncio.to_thread(
                self.extract_embedding_from_buffer, 
                audio_int16, 
                self.sample_rate
            )
            
            if ecapa_embedding is None:
                return {"error": "Failed to extract embedding"}
            
            # Find best speaker match
            #speaker_name, uid, confidence = self.ecapa_matcher.find_best_match(ecapa_embedding)
            
            # Find best speaker match WITH nomatch data (always use enhanced method)
            speaker_name, uid, confidence, nomatch_data = self.ecapa_matcher.find_best_match_with_nomatch_data(ecapa_embedding)
            # Calculate nomatch score for all extractions
            nomatch_score = self.calculate_nomatch_score(nomatch_data, buffer_duration)
     
            # Determine speaker identification result
            if confidence < self.UNCERTAIN_THRESHOLD:
                speaker_result = "unknown speaker"
                speaker_confidence = "uncertain"
            elif confidence < self.CERTAIN_THRESHOLD:
                speaker_result = f"{speaker_name}(?)"
                speaker_confidence = "uncertain"
                self.subsequent_nomatch = 0
            else:
                speaker_result = f"{speaker_name}"
                speaker_confidence = "certain"
                self.subsequent_nomatch = 0
                # We have a (near)certain match, let's cumulatively enrich the embedding for that user
                # NOTE this involves i/o and can add up quickly for large databases may need a queueing system in the future
                # Only update embedding on final utterance (when silence triggers extraction)
                if reason == "silence":
                    try:
                        success = await self.update_speaker_imprint_from_buffer(uid, ecapa_embedding, audio_int16)
                        if success:
                            print(f"[ECAPA] Successfully updated imprint for {speaker_name}")
                        else:
                            print(f"[ECAPA] Failed to update imprint for {speaker_name}")
                    except Exception as e:
                        print(f"[ECAPA] Error updating imprint: {e}")
            
            if speaker_confidence == "certain":
                print(f"[ECAPA] Speaker match result: {speaker_result} (confidence: {confidence:.3f}, {speaker_confidence})")
            
            # Update extraction tracking if this was a scheduled extraction
            if reason == "scheduled":
                self.extraction_count += 1
                self.last_extraction_bytes = len(audio_buffer)
                print(f"[ECAPA] nomatch score: {nomatch_score}")
            # Update subsequent/total nomatch if nomatch score was somewhat reliable
            if reason == "silence" and nomatch_score >= self.nomatch_lower_threshold:
                self.subsequent_nomatch += 1
                self.total_nomatch += 1
                print(f"[ECAPA] Subsequent reliable nomatch count: {self.subsequent_nomatch}")
                print(f"[ECAPA] Total reliable nomatch count: {self.total_nomatch}")
            # Initiate enrollment if nomatch score is very reliable
            if nomatch_score >= self.nomatch_upper_threshold:
                print(f"[ECAPA] High nomatch confidence detected - consider triggering enrollment flow")
                self.subsequent_nomatch = 0
                self.total_nomatch = 0
                # initiate rasa enrollment story
                suggest_enrollment = True
            # Initiate enrollment if 2 subsequent reasonably reliable utterances
            elif self.subsequent_nomatch >= 2:
                print(f"[ECAPA] 2 subsequent reasonable nomatch utterances - consider triggering enrollment flow")
                self.subsequent_nomatch = 0
                self.total_nomatch = 0
                suggest_enrollment = True
            # Initiate enrollment if 3 total reasonably reliable utterances
            elif self.total_nomatch >=3:
                print(f"[ECAPA] 3 total reasonable nomatch utterances - consider triggering enrollment flow")
                self.subsequent_nomatch = 0
                self.total_nomatch = 0
                suggest_enrollment = True
            else:
                suggest_enrollment = False

            result = {
                "speaker_name": speaker_name,
                "uid": uid,
                "confidence": confidence,
                "speaker_confidence": speaker_confidence,
                "speaker_result": speaker_result,
                "buffer_duration": buffer_duration,
                "extraction_reason": reason,
                "extraction_count": self.extraction_count + 1, # (starts at 0)
                "nomatch_score": nomatch_score,
                "nomatch_data": nomatch_data,  # Included for debugging if needed
                "suggest_enrollment": suggest_enrollment
            }
            
            return result
            
        except Exception as e:
            print(f"[ECAPA] Error in extract_and_match_from_buffer: {e}")
            return {"error": str(e)}

class RasaClient:
    """
    Handles communication with Rasa server for intent recognition and response generation.
    """
    
    def __init__(self, rasa_url: str, timeout: int = 10):
        self.rasa_url = rasa_url.rstrip('/')
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def send_message(self, message, client_id, speaker_name=None):
        """
        Send a message to Rasa and get the response.
        
        Args:
            message (str): The user's message/transcript
            sender (str): Sender identifier (default: "user")
            
        Returns:
            Dict containing Rasa response or None if failed
        """
        if not self.session:
            print("[Rasa] Error: Session not initialized. Use async context manager.")
            return None
            
        try:
            payload = {
                "sender": client_id,
                "message": message
            }

            if speaker_name:
                payload["metadata"] = {"speaker_name": speaker_name}
            
            print(f"[Rasa] Sending message: '{message}'")
            
            async with self.session.post(
                f"{self.rasa_url}/webhooks/rest/webhook",
                json=payload
            ) as response:
                
                if response.status == 200:
                    rasa_response = await response.json()
                    print(f"[Rasa] Received response: {rasa_response}")
                    return rasa_response
                else:
                    print(f"[Rasa] HTTP Error {response.status}: {await response.text()}")
                    return None
                    
        except asyncio.TimeoutError:
            print(f"[Rasa] Timeout after {self.timeout} seconds")
            return None
        except aiohttp.ClientError as e:
            print(f"[Rasa] Client error: {e}")
            return None
        except Exception as e:
            print(f"[Rasa] Unexpected error: {e}")
            return None

async def process_rasa_response(client_id: str, rasa_response: list) -> bool:
    """
    Process Rasa response and send appropriate messages/audio to client.
    
    Args:
        client_id (str): WebSocket client ID
        rasa_response (list): List of response objects from Rasa
        speaker (str): Speaker name for TTS (optional)
        
    Returns:
        bool: True if any responses were processed
    """
    if not rasa_response:
        print("[Rasa] No response from Rasa")
        return False
    
    processed_any = False
    server_name = CONFIG.get("server_name", "Fawkes")
    
    for response_item in rasa_response:
        if "text" in response_item:
            response_text = response_item["text"]
            print(f"[Rasa] Processing text response: '{response_text}'")
            
            # Send transcript message to client
            data_to_send = {
                "speaker": server_name,
                "speaker_confidence": "certain",
                "final": "True",
                "transcript": response_text,
                "asr_confidence": "certain"
            }
            json_string = json.dumps(data_to_send)
            await send_message_to_client(client_id, json_string)
            
            # Handle TTS if not using client-side TTS
            if not clientSideTTS and active_websockets:
                #print(f"[Rasa] Using Piper TTS")
                asyncio.create_task(stream_tts_audio(client_id, response_text))
            
            processed_any = True
        
        elif "custom" in response_item:
            # Handle custom responses (e.g., actions, data)
            custom_data = response_item["custom"]
            print(f"[Rasa] Processing custom response: {custom_data}")
            # Add custom handling logic here as needed
            processed_any = True
    
    return processed_any

async def handle_final_utterance_with_rasa(client_id, final_transcription_text, speaker_name):
    """
    Process final utterance through Rasa instead of hardcoded logic.
    
    Args:
        client_id (str): WebSocket client ID
        final_transcription_text (str): The final transcribed text
        speaker (str): Identified speaker
    """
    if not CONFIG.get("enable_rasa", False):
        print("[Rasa] Rasa integration disabled, skipping...")
        return
    
    if not final_transcription_text.strip():
        print("[Rasa] Empty transcription, skipping Rasa processing")
        return
    
    try:
        async with RasaClient(CONFIG["rasa_url"], CONFIG["rasa_timeout"]) as rasa_client:
            # Send the transcription to Rasa
            rasa_response = await rasa_client.send_message(
                final_transcription_text, 
                sender=f"client_{client_id}",
                speaker_name=speaker_name
            )
            
            # Process the response
            if rasa_response:
                success = await process_rasa_response(
                    client_id, 
                    rasa_response
                )
                if not success:
                    print("[Rasa] No valid responses to process")
            else:
                print("[Rasa] Failed to get response from Rasa")
                
    except Exception as e:
        print(f"[Rasa] Error in handle_final_utterance_with_rasa: {e}")

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

async def send_message_to_client(client_id, message):
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

async def save_utterance_async(audio_bytes: bytes) -> str:
    """
    Saves a complete audio utterance to a 16-bit, 16kHz, mono WAV file
    in a non-blocking manner.

    This function uses `asyncio.to_thread()` to run the file I/O
    in a separate thread, preventing it from blocking the main event loop.

    Args:
        audio_bytes (bytes): The complete audio utterance as bytes.

    Returns:
        str: The full path to the saved file.
    """
    # Define the target audio parameters
    OUTPUT_DIR = "utterances"
    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2 # 16-bit
    NUM_CHANNELS = 1
    # Generate a unique filename using a UUID
    filename = f"utterance_{uuid.uuid4()}.wav"
    filepath = f"{OUTPUT_DIR}/{filename}"

    # Use a thread pool executor to run the blocking I/O operation
    def _save_file():
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(NUM_CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_bytes)
            print(f"Successfully saved utterance to {filepath}")
        except Exception as e:
            # If the directory doesn't exist, this will raise a FileNotFoundError
            print(f"Error saving utterance to {filepath}: {e}")
            raise e # Re-raise the exception

    # Run the blocking file-saving function in a separate thread
    await asyncio.to_thread(_save_file)
    
    return filepath

async def manual_sequential_ecapa(firstname, surname, update_wav_paths):
    """
    Perform sequential ECAPA embedding updates for a specific speaker.
    
    Args:
        firstname (str): First name of the speaker
        surname (str): Last name of the speaker (can be None)
        update_wav_paths (list): List of WAV file paths for updating the speaker's embedding
        
    Returns:
        dict: Results summary with success/failure information
    """
    print(f"\n--- Sequential ECAPA updates for {firstname} {surname if surname else ''} ---")
    
    # Query to get the speaker's UID
    if surname:
        uid_result = con.execute("""
            SELECT uid FROM speakers 
            WHERE firstname = ? AND surname = ?
        """, (firstname, surname)).fetchone()
        speaker_display_name = f"{firstname} {surname}"
    else:
        uid_result = con.execute("""
            SELECT uid FROM speakers 
            WHERE firstname = ? AND surname IS NULL
        """, (firstname,)).fetchone()
        speaker_display_name = firstname
    
    if not uid_result:
        error_msg = f"Error: Could not find speaker '{speaker_display_name}' in database"
        print(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "speaker": speaker_display_name,
            "updates_attempted": 0,
            "updates_successful": 0
        }
    
    speaker_uid = uid_result[0]
    print(f"Found {speaker_display_name} with UID: {speaker_uid}")
    
    # Get initial metadata
    initial_metadata = con.execute("""
        SELECT total_duration_sec, sample_count 
        FROM speakers 
        WHERE uid = ?
    """, (speaker_uid,)).fetchone()
    
    initial_duration, initial_count = initial_metadata if initial_metadata else (0.0, 0)
    print(f"Initial state - Duration: {initial_duration:.2f}s, Sample count: {initial_count}")
    
    # Process updates sequentially
    successful_updates = 0
    failed_updates = []
    
    try:
        for i, wav_path in enumerate(update_wav_paths, 1):
            print(f"\nProcessing update {i}/{len(update_wav_paths)}: {Path(wav_path).name}")
            
            try:
                success = await ecapa_processor.update_speaker_imprint_from_file(wav_path, speaker_uid)
                if success:
                    successful_updates += 1
                    print(f"✓ Update {i} completed successfully")
                else:
                    failed_updates.append({"index": i, "path": wav_path, "error": "Function returned False"})
                    print(f"✗ Update {i} failed - function returned False")
                    
            except FileNotFoundError:
                failed_updates.append({"index": i, "path": wav_path, "error": "File not found"})
                print(f"✗ Update {i} failed - file not found: {wav_path}")
            except Exception as e:
                failed_updates.append({"index": i, "path": wav_path, "error": str(e)})
                print(f"✗ Update {i} failed - error: {e}")
                
    except Exception as e:
        error_msg = f"Critical error during sequential updates: {e}"
        print(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "speaker": speaker_display_name,
            "updates_attempted": len(update_wav_paths),
            "updates_successful": successful_updates,
            "failed_updates": failed_updates
        }
    
    # Get final metadata
    final_metadata = con.execute("""
        SELECT total_duration_sec, sample_count, last_updated 
        FROM speakers 
        WHERE uid = ?
    """, (speaker_uid,)).fetchone()
    
    if final_metadata:
        final_duration, final_count, last_update = final_metadata
        print(f"\n--- Final metadata for {speaker_display_name} ---")
        print(f"Total duration: {final_duration:.2f}s (+{final_duration - initial_duration:.2f}s)")
        print(f"Sample count: {final_count} (+{final_count - initial_count})")
        print(f"Last updated: {last_update}")
    
    # Summary
    print(f"\n--- Update Summary ---")
    print(f"Speaker: {speaker_display_name}")
    print(f"Updates attempted: {len(update_wav_paths)}")
    print(f"Updates successful: {successful_updates}")
    print(f"Updates failed: {len(failed_updates)}")
    
    if failed_updates:
        print("Failed updates:")
        for failure in failed_updates:
            print(f"  - Update {failure['index']}: {Path(failure['path']).name} ({failure['error']})")
    
    return {
        "success": successful_updates > 0,
        "speaker": speaker_display_name,
        "speaker_uid": speaker_uid,
        "updates_attempted": len(update_wav_paths),
        "updates_successful": successful_updates,
        "updates_failed": len(failed_updates),
        "failed_updates": failed_updates,
        "initial_metadata": {"duration": initial_duration, "count": initial_count},
        "final_metadata": {"duration": final_duration, "count": final_count, "last_updated": last_update} if final_metadata else None
    }

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
    SPEAKER = CONFIG["default_speaker"]
    SPEAKER_CONFIDENCE = CONFIG["default_speaker_confidence"]
    ASR_CONFIDENCE = CONFIG["default_asr_confidence"]
    SERVER = CONFIG["server_name"]

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
                            # Reset ECAPA processor for new utterance
                            ecapa_processor.reset_for_new_utterance()

                        # Perform ASR transcription on the *current audio chunk* if speech is active
                        text = await asyncio.to_thread(nemo_transcriber.transcribe_chunk, audio_chunk_np)
                        final_transcription_text = text  # Keep updating the final transcription

                        # Check if we should extract ECAPA embedding
                        if ecapa_processor.should_extract_now(len(current_utterance_buffer)):
                            ecapa_result = await ecapa_processor.extract_and_match_from_buffer(
                                current_utterance_buffer, 
                                reason="scheduled"
                            )
                            # You can log or process the ecapa_result as needed
                            if "error" not in ecapa_result:
                                print(f"[Speaker ID] {ecapa_result['speaker_result']}")
                                SPEAKER = ecapa_result['speaker_result']
                                SPEAKER_CONFIDENCE = ecapa_result['speaker_confidence']

                        data_to_send = {
                            "speaker": SPEAKER,
                            "speaker_confidence": SPEAKER_CONFIDENCE,
                            "final": False, # Always interim while speaking
                            "transcript": text,
                            "asr_confidence": ASR_CONFIDENCE
                        }
                        json_string = json.dumps(data_to_send)
                        await send_message_to_client(client_id, json_string)

                    else: # VAD indicates silence
                        silence_counter += 1
                        if is_speaking and silence_counter >= SILENCE_CHUNKS_THRESHOLD:
                            #is_speaking = False
                            #print(f"[{client_id}] Voice activity ended. Processing final utterance.")
                            print("Acoustic finality detected. Processing full utterance with offline model...")
                            # UNCOMMENT TO SAVE EACH UTTERANCE TO WAV
                            #asyncio.create_task(save_utterance_async(current_utterance_buffer))

                            # Extract final ECAPA embedding before clearing buffer
                            if len(current_utterance_buffer) > 0:
                                final_ecapa_result = await ecapa_processor.extract_and_match_from_buffer(
                                    current_utterance_buffer,
                                    reason="silence"
                                )
                                if "error" not in final_ecapa_result:
                                    print(f"[Final Speaker ID] {final_ecapa_result['speaker_result']}")
                                    SPEAKER = final_ecapa_result['speaker_result']
                                    SPEAKER_CONFIDENCE = final_ecapa_result['speaker_confidence']

                            # We now have accoustic finality. Perform final offline n-best beam search
                            # Then send to rescorer and P&C to determine linguistic finality

                            # Convert the accumulated utterance buffer to numpy array
                            # The buffer is already in 16kHz 16-bit mono PCM format from the client
                            '''
                            audio_int16 = np.frombuffer(current_utterance_buffer, dtype=np.int16)

                            # Use Canary-Qwen for final transcription with ITN and P&C
                            # Pass the int16 audio directly - no need for float32 conversion here
                            # since the transcriber handles that internally
                            
                            final_canary_transcription = await asyncio.to_thread(
                                canary_qwen_transcriber.transcribe_final, 
                                audio_int16,
                                CONFIG["audio_sample_rate"]  # Should be 16000
                            )
                            print(f"Canary-Qwen final transcription: '{final_canary_transcription}'")
                            
                            # Use Canary-Qwen result as the final transcription if available
                            # Otherwise fall back to the streaming result
                            if final_canary_transcription.strip():
                                final_transcription_text = final_canary_transcription
                            '''
                            # else keep the existing final_transcription_text from streaming
                            if final_transcription_text:
                                data_to_send = {
                                    "speaker": SPEAKER,
                                    "speaker_confidence": SPEAKER_CONFIDENCE,
                                    "final": True,
                                    "transcript": final_transcription_text,
                                    "asr_confidence": ASR_CONFIDENCE
                                }
                                json_string = json.dumps(data_to_send)
                                await send_message_to_client(client_id, json_string)

                                # Send final utterance to Rasa for intent identification
                                await handle_final_utterance_with_rasa(client_id, final_transcription_text, SPEAKER)

                            # Reset the buffer and state for the next utterance
                            is_speaking = False
                            silence_counter = 0
                            current_utterance_buffer = b''
                            #previous_text = "" # Reset previous_text for the new utterance
                            text = ""
                            # Reset ECAPA processor
                            SPEAKER = CONFIG["default_speaker"]
                            SPEAKER_CONFIDENCE = CONFIG["default_speaker_confidence"]
                            #ASR_CONFIDENCE = CONFIG["default_asr_confidence"]
                            ecapa_processor.reset_for_new_utterance()

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
    global main_loop, pipertts_wrapper, xtts_wrapper, nemo_vad
    global nemo_transcriber, canary_qwen_transcriber
    global ecapa_matcher, ecapa_processor
    xtts_wrapper = canary_qwen_transcriber = None # These are often turned off
    main_loop = asyncio.get_event_loop()  # Store the event loop
    # Setup the database table before starting the server
    setup_database()
    # Initialize PiperTTS instance
    pipertts_wrapper = PiperTTS(CONFIG["piper_model_path"])
    # Initialize Coqui XTTS instance
    xtts_wrapper = XTTSWrapper(
        CONFIG["xtts_model_dir"],
        CONFIG["inference_device"],
        CONFIG["speakers_dir"]
    )

    # Initialize the NeMo VAD model
    nemo_vad = NeMoVAD(
        model_path=CONFIG["nemo_vad_model_path"],
        device=CONFIG["inference_device"],
        sample_rate=CONFIG["vad_sample_rate"]
    )
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
    # Load in-memory ECAPA matching routine
    ecapa_matcher = FastECAPASpeakerMatcher(con)
    # Initialize unified ECAPA processor (add this after ecapa_matcher)
    ecapa_processor = ECAPASpeakerProcessor(
        model_path=CONFIG["ecapa_tdnn_model_path"],
        device=CONFIG["inference_device"],
        ecapa_matcher=ecapa_matcher,
        sample_rate=CONFIG["audio_sample_rate"]
    )

    # ADD INITIAL SPEAKERS TO DB
    '''
    # 1. Query the database to see how many speakers are there before adding
    print("--- Number of speakers BEFORE adding new speaker ---")
    results_before = con.execute("SELECT COUNT(*) FROM speakers").fetchall()
    print(f"Current speaker count: {results_before[0][0]}")
    # 2. Call the extract_embed function
    wav_path_1 = "/root/fawkes/audio_samples/_preprocessed/nate_jabra_mic.wav"
    wav_path_2 = "/root/fawkes/audio_samples/_preprocessed/courtney_02.wav"
    wav_path_3 = "/root/fawkes/audio_samples/_preprocessed/neilgaiman_01.wav"
    try:
        #await xtts_wrapper.extract_xtts_embed(wav_path, firstname="neil", surname="gaiman")
        #await ecapa_processor.create_initial_speaker_imprint(wav_path, firstname="nathanael", surname="warren")
        results = await asyncio.gather(
            ecapa_processor.create_initial_speaker_imprint(wav_path_1, firstname="nathanael", surname="warren"),
            ecapa_processor.create_initial_speaker_imprint(wav_path_2, firstname="courtney", surname="mosierwarren"),
            ecapa_processor.create_initial_speaker_imprint(wav_path_3, firstname="neil", surname="gaiman")
        )
    except FileNotFoundError:
        print(f"Error: The file(s) were not found. Please check the paths and try again.")
    except Exception as e:
        print(f"An error occurred while adding the speaker: {e}")
    # 3. Query the database again to see if the speaker was added successfully
    print("\n--- Speakers AFTER adding new speaker ---")
    results_after = con.execute("SELECT firstname, surname FROM speakers").fetchall()
    print("Updated speaker list:")
    for row in results_after:
        print(f"- {row[0]} {row[1]}")
    # Re-load the speakers into the model's speaker manager to make the new speaker available
    xtts_wrapper._load_speakers_from_db()
    '''

    # ADD SEQUENTIAL ECAPA UPDATES MANUALLY
    '''
    nathanael_wavs = [
        "/root/fawkes/audio_samples/_preprocessed/nathanael_01.wav",
        "/root/fawkes/audio_samples/_preprocessed/nathanael_02.wav", 
        "/root/fawkes/audio_samples/_preprocessed/nate_iphone_mic.wav",
        "/root/fawkes/audio_samples/_preprocessed/nate_samson_meteorite.wav"
    ]
    result = await manual_sequential_ecapa("nathanael", "warren", nathanael_wavs)
    courtney_wavs = [
        "/root/fawkes/audio_samples/_preprocessed/courtney_01.wav"
    ]
    result = await manual_sequential_ecapa("courtney", "mosierwarren", courtney_wavs)
    '''
    
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
