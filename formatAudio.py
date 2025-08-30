import os
from pydub import AudioSegment

def format_audio_files(source_dir, output_dir):
    """
    Rewrites all .wav files in a directory to a consistent format.

    The files are resampled to 16 kHz, converted to mono, and saved as 16-bit PCM.
    This format is required for many deep learning models, including the
    ECAPA-TDNN speaker embedding model.

    Args:
        source_dir (str): The directory containing the original .wav files.
        output_dir (str): The directory where the processed .wav files will be saved.
                          This directory will be created if it does not exist.
    """
    # Create the output directory if it doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Process each file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".wav"):
            source_path = os.path.join(source_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                # Load the audio file
                audio = AudioSegment.from_wav(source_path)

                # Set the required parameters for the ECAPA model:
                # - Frame rate (sample rate) to 16000 Hz
                # - Channels to 1 (mono)
                # - Sample width to 2 bytes (16-bit signed PCM)
                audio = audio.set_frame_rate(16000)
                audio = audio.set_channels(1)
                audio = audio.set_sample_width(2)

                # Export the formatted audio to the new directory
                audio.export(output_path, format="wav")
                print(f"Successfully processed and saved: {filename}")
            except Exception as e:
                print(f"Failed to process {filename}. Error: {e}")

if __name__ == "__main__":
    # Define the source directory where your audio files are located
    SOURCE_DIRECTORY = '/root/fawkes/audio_samples'
    
    # Define the output directory where the formatted files will be saved
    OUTPUT_DIRECTORY = os.path.join(SOURCE_DIRECTORY, '_preprocessed')

    print("Starting audio file formatting...")
    format_audio_files(SOURCE_DIRECTORY, OUTPUT_DIRECTORY)
    print("Formatting complete.")
    print(f"All processed files can be found in: {OUTPUT_DIRECTORY}")

