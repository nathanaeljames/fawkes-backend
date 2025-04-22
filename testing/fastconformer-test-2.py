import time
import torch
import torchaudio
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from omegaconf import OmegaConf
from pathlib import Path
import argparse
import numpy as np

# Constants
SAMPLE_RATE = 16000

def load_audio(file_path):
    audio, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        audio = resampler(audio)
    return audio.mean(dim=0, keepdim=True)  # Convert to mono

def split_audio(audio, chunk_ms):
    chunk_samples = int((chunk_ms / 1000) * SAMPLE_RATE)
    return [audio[:, i:i+chunk_samples] for i in range(0, audio.shape[1], chunk_samples)]

def benchmark_chunks(model, chunks):
    import time

    total_duration = 0.0
    total_time = 0.0

    for i, chunk in enumerate(chunks):
        if chunk.ndim > 1:
            chunk = chunk.squeeze()
        if not isinstance(chunk, np.ndarray):
            chunk = np.array(chunk, dtype=np.float32)
        elif chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        duration = len(chunk) / 16000
        start = time.perf_counter()
        _ = model.transcribe([chunk])[0]
        elapsed = time.perf_counter() - start

        total_duration += duration
        total_time += elapsed
        print(f"Chunk {i+1:02d} - Duration: {duration:.2f}s | Inference Time: {elapsed:.2f}s | RT Factor: {elapsed/duration:.2f}")

    print(f"\nTotal Audio: {total_duration:.2f}s | Total Inference Time: {total_time:.2f}s | Overall RT Factor: {total_time/total_duration:.2f}")

def main(audio_path, model_path, chunk_ms):
    print(f"\n🔍 Loading model from {model_path}")
    #model = EncDecRNNTBPEModel.restore_from(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    #model = EncDecHybridRNNTCTCBPEModel.restore_from("/root/fawkes/models/fc-hybrid-lg-multi/stt_en_fastconformer_hybrid_large_streaming_multi.nemo")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncDecHybridRNNTCTCBPEModel.restore_from(model_path, map_location=device)
    if not hasattr(model.cfg.decoder, "streaming_cfg"):
        model.cfg.decoder.streaming_cfg = OmegaConf.create()
    # 🎯 Hardcoded context window values (in frames)
    # NOTE: 1 frame = typically 10ms of audio
    model.cfg.decoder.streaming_cfg.left_context = 128     # ~1.28 sec memory
    model.cfg.decoder.streaming_cfg.right_context = 32     # ~320ms look-ahead
    model.cfg.decoder.streaming_cfg.chunk_size = 160       # ~1.6 sec processing window (optional, can tune)

    print("Streaming config updated:")
    print("  Left context:", model.cfg.decoder.streaming_cfg.left_context)
    print("  Right context:", model.cfg.decoder.streaming_cfg.right_context)
    print("  Chunk size:", model.cfg.decoder.streaming_cfg.chunk_size)
    model.eval()

    print(f"🎧 Loading audio from {audio_path}")
    audio = load_audio(audio_path)
    chunks = split_audio(audio, chunk_ms)

    print(f"🧩 Split into {len(chunks)} chunks of {chunk_ms} ms")
    benchmark_chunks(model, chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", type=str, help="Path to WAV file")
    parser.add_argument("--model_path", type=str, default="/root/fawkes/models/fc-hybrid-lg-multi/stt_en_fastconformer_hybrid_large_streaming_multi.nemo")
    parser.add_argument("--chunk_ms", type=int, default=1000, help="Chunk size in milliseconds (default: 1000ms)")
    args = parser.parse_args()

    main(args.wav_path, args.model_path, args.chunk_ms)
