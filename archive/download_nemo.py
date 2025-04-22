from nemo.collections.asr.models import EncDecCTCModelBPE

model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_citrinet_1024_streaming")
model.save_to("/root/fawkes/models/nemo_asr/stt_en_citrinet_1024_streaming.nemo")
