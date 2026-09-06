# minimal + use local model & run off gpu
'''
branch in two directions:
    transcribe wav file with interim and final results/ utilize FrameBatchASR
    transcribe existing streaming buffer with simple print
'''

import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecRNNTBPEModel
#from nemo.collections.asr.parts.streaming.frame_batch_asr import FrameBatchASR
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR

device = "cuda" if torch.cuda.is_available() else "cpu"
if (device != "cuda"):
    print('Warning! GPU not detected!')

# specify ENCODER_STEP_LENGTH (which is 80 ms for FastConformer models)
#ENCODER_STEP_LENGTH = 80 # ms

#asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")
MODEL_PATH = "/root/fawkes/models/fc-hybrid-lg-multi/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
asr_model = EncDecRNNTBPEModel.restore_from(MODEL_PATH, map_location=torch.device(device))

# Optional: change the default latency. Default latency is 1040ms. Supported latencies: {0: 0ms, 1: 80ms, 16: 480ms, 33: 1040ms}.
# Note: These are the worst latency and average latency would be half of these numbers.
# Supported values for att_context_size: {[70,0]: 0ms, [70,1]: 80ms, [70,16]: 480ms, [70,33]: 1040ms}.
#asr_model.encoder.set_default_att_context_size([70,13]) 
asr_model.encoder.set_default_att_context_size([70,16]) 

#Optional: change the default decoder. Default decoder is Transducer (RNNT). Supported decoders: {ctc, rnnt}.
asr_model.change_decoding_strategy(decoder_type='rnnt')

output = asr_model.transcribe(['transcribe/16khz_example_speech.wav'])
print(output[0].text)