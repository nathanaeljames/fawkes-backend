import nemo.collections.asr as nemo_asr

all_asr_models = nemo_asr.models.EncDecClassificationModel.list_available_models()
marblenet_models = [model.pretrained_model_name for model in all_asr_models if 'marblenet' in model.pretrained_model_name.lower()]

for model_name in marblenet_models:
    print(model_name)