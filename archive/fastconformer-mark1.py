# absolute minimum implementation of fastconformer model
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")

# Optional: change the default latency. Default latency is 1040ms. Supported latencies: {0: 0ms, 1: 80ms, 16: 480ms, 33: 1040ms}.
# Note: These are the worst latency and average latency would be half of these numbers.
# Supported values for att_context_size: {[70,0]: 0ms, [70,1]: 80ms, [70,16]: 480ms, [70,33]: 1040ms}.
asr_model.encoder.set_default_att_context_size([70,13]) 

#Optional: change the default decoder. Default decoder is Transducer (RNNT). Supported decoders: {ctc, rnnt}.
asr_model.change_decoding_strategy(decoder_type='rnnt')

output = asr_model.transcribe(['transcribe/16khz_example_speech.wav'])
print(output[0].text)