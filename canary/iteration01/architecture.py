# High-level ASR Streaming System Architecture with Speaker Features

# Modules:
# 1. Audio Ingestion & Preprocessing
# 2. Streaming ASR Model (FastConformer)
# 3. Interim + Final Decoder
# 4. Speaker Diarization (pyannote-audio)
# 5. Speaker Embedding & Recognition
# 6. Post-processing: Contextual Rescoring
# 7. Output Streaming

# ----------------------------------------------------------------------------

# Module 1: Audio Ingestion (16kHz PCM stream to buffer)
async def audio_ingest(websocket, client_id):
    while True:
        chunk = await websocket.receive_bytes()
        await client_queues[client_id]["incoming_audio"].put(chunk)

# ----------------------------------------------------------------------------

# Module 2: Streaming ASR (FastConformer + CTC interim)
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR

asr_model = EncDecRNNTBPEModel.restore_from("nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")
streaming_decoder = FrameBatchASR(
    asr_model,
    frame_len=6400,  # 0.4s chunks (assuming 16kHz)
    total_buffer=25600,
    pad_align=True,
)

async def run_streaming_asr(client_id):
    buffer = b""
    while True:
        chunk = await client_queues[client_id]["incoming_audio"].get()
        buffer += chunk
        if len(buffer) >= 6400:
            result = streaming_decoder.transcribe([buffer])
            await client_queues[client_id]["raw_asr"].put(result)
            buffer = b""

# ----------------------------------------------------------------------------

# Module 3: Decode Interim + Final Results
async def decode_asr_output(client_id):
    while True:
        result = await client_queues[client_id]["raw_asr"].get()
        # CTC hypothesis for interim
        interim_text = result[0]['ctc']
        final_text = result[0]['rnnt']
        confidence = result[0].get('confidence', 0.0)
        await client_queues[client_id]["outgoing_text"].put({
            "interim": interim_text,
            "final": final_text,
            "confidence": confidence,
        })

# ----------------------------------------------------------------------------

# Module 4: Speaker Diarization (run in parallel)
from pyannote.audio import Pipeline
speaker_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

async def run_diarization(client_id):
    # Buffer to file or memory
    with open(f"buffer_{client_id}.wav", "wb") as f:
        while True:
            chunk = await client_queues[client_id]["incoming_audio"].get()
            f.write(chunk)
            if diarization_ready():
                break
    diarization_result = speaker_pipeline(f"buffer_{client_id}.wav")
    await client_queues[client_id]["diarization"].put(diarization_result)

# ----------------------------------------------------------------------------

# Module 5: Speaker Recognition + Embedding Matching
from pyannote.audio import Inference
speaker_embedder = Inference("pyannote/embedding")
known_speakers = load_known_speaker_embeddings()

def match_speaker(audio_chunk):
    embedding = speaker_embedder(audio_chunk)
    return compare_to_known(embedding, known_speakers)

# ----------------------------------------------------------------------------

# Module 6: Contextual Rescoring (Optional)
from transformers import AutoModelForCausalLM, AutoTokenizer
lm_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def rescore(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = lm_model.generate(**inputs, max_new_tokens=0, return_dict_in_generate=True)
    return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

# ----------------------------------------------------------------------------

# Module 7: Stream Output to WebSocket
async def send_transcription(websocket, client_id):
    while True:
        text = await client_queues[client_id]["outgoing_text"].get()
        await websocket.send_json(text)

# ----------------------------------------------------------------------------

# Note: each module can be run as a coroutine using asyncio.create_task
# This allows full-duplex streaming of audio, transcription, and speaker data.

'''
Streaming Input (audio chunks)
      ↓
RNNT Model (with CTC interim outputs)
      ↓
Interim results → Send to UI (optional, unformatted)
      ↓
Buffer full utterance
      ↓
→ Final RNNT output (context-rich, higher confidence)
      ↓
→ Contextual Rescoring (with optional LM, n-best reranking)
      ↓
→ Final utterance (lowercased, raw)
      ↓
→ Punctuation + Capitalization module
      ↓
→ Prettified final text → Send to UI
'''

'''
ÜBER STT

Ideally, completion detection would take into consideration much more that "silence period" or "frames without change" [in transcription]
Finality detection should consider intonation, speech rate, prosody, grammatical completeness, turn-taking, etc
Much of this is requires direct access to speech data, thus finality detection AT MINIMUM should be done by the ASR model
Commas may also benefit from direct speech data, while most other punctuation is grammatically determined

Other considerations of a robust ASR model/framework (which would benefit greatly from direct speech data):
Finality detection
Irrelevance detection
Turn taking
Interruption and resumption

These may benefit from running on the same model OR could be additions to a pipeline but should really have access to 
raw audio data and processing it could potentially represent duplication of work

Many speaker diaization models implement very robust VAD, and I can use this in the future to signal finality even
during simultaneous speaker events
'''
