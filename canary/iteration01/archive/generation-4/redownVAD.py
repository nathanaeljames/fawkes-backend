import nemo.collections.asr as nemo_asr
import os

# Define the local path where you want to save the model
# Make sure this path exists or create it before running the script
LOCAL_VAD_MODEL_PATH = "/root/fawkes/models/marblenet_vad_multi/frame_vad_multilingual_marblenet_v2.0.nemo"
#LOCAL_VAD_MODEL_PATH = "/root/fawkes/models/marblenet_vad_multi/frame_vad_multilingual_marblenet_latest.nemo"

# Ensure the directory exists
os.makedirs(os.path.dirname(LOCAL_VAD_MODEL_PATH), exist_ok=True)

print(f"Attempting to download and save vad_multilingual_marblenet to {LOCAL_VAD_MODEL_PATH}...")

try:
    # Download the model from NGC
    vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name="vad_multilingual_marblenet")

    # Save the model to the specified local path
    vad_model.save_to(LOCAL_VAD_MODEL_PATH)

    print(f"Successfully downloaded and saved vad_multilingual_marblenet to: {LOCAL_VAD_MODEL_PATH}")

except Exception as e:
    print(f"Error downloading or saving the VAD model: {e}")
    print("Please ensure you have an active internet connection and sufficient disk space.")