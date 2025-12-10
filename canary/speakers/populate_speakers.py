"""
Speaker Population Script
This script loads the AI models and populates the speakers table with initial speaker data.
Requires: XTTS model, ECAPA-TDNN model, and audio samples

WARNING: This script loads heavy AI models and may take several minutes to run.
"""

import asyncio
import duckdb
import torch
import json
import librosa
import numpy as np
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import nemo.collections.asr as nemo_asr

# Configuration - UPDATE THESE PATHS TO MATCH YOUR SETUP
CONFIG = {
    "duckdb_path": "./database.duckdb",
    "xtts_model_dir": "/root/fawkes/models/coqui_xtts/XTTS-v2/",
    "ecapa_tdnn_model_path": "/root/fawkes/models/ecapa_tdnn_embed/ecapa_tdnn.nemo",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sample_rate": 16000,
}

# Initial speakers to add - UPDATE THESE PATHS TO YOUR AUDIO SAMPLES
INITIAL_SPEAKERS = [
    {
        "firstname": "Nathanael",
        "surname": "Warren", 
        "wav_path": "/root/fawkes/audio_samples/_preprocessed/nate_jabra_mic.wav"
    },
    {
        "firstname": "Courtney",
        "surname": "Mosier Warren",
        "wav_path": "/root/fawkes/audio_samples/_preprocessed/courtney_02.wav"
    },
    {
        "firstname": "Neil",
        "surname": "Gaiman",
        "wav_path": "/root/fawkes/audio_samples/_preprocessed/neilgaiman_01.wav"
    }
]

# Sequential updates - Additional audio samples to improve existing speaker embeddings
SPEAKER_UPDATES = [
    {
        "firstname": "Nathanael",
        "surname": "Warren",
        "wav_paths": [
            "/root/fawkes/audio_samples/_preprocessed/nathanael_01.wav",
            "/root/fawkes/audio_samples/_preprocessed/nathanael_02.wav",
            "/root/fawkes/audio_samples/_preprocessed/nate_iphone_mic.wav",
            "/root/fawkes/audio_samples/_preprocessed/nate_samson_meteorite.wav"
        ]
    },
    {
        "firstname": "Courtney",
        "surname": "Mosier Warren",
        "wav_paths": [
            "/root/fawkes/audio_samples/_preprocessed/courtney_01.wav"
        ]
    }
]


class SimpleXTTSWrapper:
    """Minimal XTTS wrapper for extracting embeddings"""
    
    def __init__(self, model_dir, device):
        print(f"Loading XTTS model from {model_dir}...")
        config = XttsConfig()
        config.load_json(f"{model_dir}/config.json")
        self.xtts_model = Xtts.init_from_config(config)
        self.xtts_model.load_checkpoint(
            config,
            checkpoint_dir=model_dir,
            use_deepspeed=False
        )
        self.xtts_model.to(device)
        self.xtts_model.eval()
        print("✓ XTTS model loaded")


class SimpleECAPAProcessor:
    """Minimal ECAPA processor for extracting embeddings"""
    
    def __init__(self, model_path, device, sample_rate=16000):
        print(f"Loading ECAPA-TDNN model from {model_path}...")
        self.device = device
        self.sample_rate = sample_rate
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
            model_path,
            map_location=torch.device(self.device)
        )
        self.model.eval()
        self.model.to(self.device)
        print("✓ ECAPA-TDNN model loaded")
    
    def extract_embedding_from_file(self, wav_path, sample_rate=None):
        """Extract ECAPA embedding from a WAV file"""
        try:
            import torchaudio
            
            if sample_rate is None:
                sample_rate = self.sample_rate
            
            # Load and preprocess audio
            audio, sr = torchaudio.load(wav_path)
            
            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                audio = resampler(audio)
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Move to device
            audio = audio.to(self.device)
            
            # Compute signal length (number of samples in the audio)
            # NeMo models expect a tensor of lengths for batched input
            audio_length = torch.tensor([audio.shape[1]], dtype=torch.long).to(self.device)
            
            # Get embedding
            with torch.no_grad():
                _, embedding = self.model(
                    input_signal=audio, 
                    input_signal_length=audio_length
                )
            
            return embedding.cpu().numpy()
            
        except Exception as e:
            print(f"Error extracting ECAPA embedding: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def update_speaker_imprint_from_file(self, con, wav_path, uid):
        """
        Update an existing speaker's ECAPA embedding using cumulative averaging.
        Simplified version of the method from server01f.py.
        """
        try:
            wav_path = Path(wav_path)
            
            # Extract new embedding
            new_embedding = self.extract_embedding_from_file(wav_path, self.sample_rate)
            if new_embedding is None:
                return False
            
            # Get current embedding and metadata from database
            result = con.execute("""
                SELECT ecapa_embedding, total_duration_sec, sample_count 
                FROM speakers 
                WHERE uid = ?
            """, (uid,)).fetchone()
            
            if not result:
                print(f"  ✗ Speaker with UID {uid} not found")
                return False
            
            current_embedding_flat, current_duration, current_count = result
            
            # Get audio duration
            try:
                new_duration = librosa.get_duration(path=str(wav_path))
            except:
                new_duration = 0.0
            
            # Perform cumulative averaging
            if current_embedding_flat and len(current_embedding_flat) > 0:
                import numpy as np
                current_embedding = np.array(current_embedding_flat).reshape(new_embedding.shape)
                
                # Weighted average based on sample count
                total_samples = current_count + 1
                updated_embedding = (
                    (current_embedding * current_count) + new_embedding
                ) / total_samples
            else:
                # No existing embedding, use new one
                updated_embedding = new_embedding
                total_samples = 1
            
            # Update database
            updated_duration = current_duration + new_duration
            updated_embedding_flat = updated_embedding.flatten().tolist()
            
            con.execute("""
                UPDATE speakers 
                SET ecapa_embedding = ?,
                    total_duration_sec = ?,
                    sample_count = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE uid = ?
            """, (updated_embedding_flat, updated_duration, total_samples, uid))
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error updating speaker imprint: {e}")
            return False


async def create_initial_speaker_imprint(
    con, xtts_wrapper, ecapa_processor, wav_path, firstname, surname=None
):
    """
    Extract both XTTS and ECAPA embeddings from a WAV file and store in database.
    This is a simplified version of the function from server01f.py.
    """
    wav_path = Path(wav_path)
    display_name = f"{firstname} {surname if surname else ''}"
    print(f"\n{'=' * 60}")
    print(f"Creating speaker imprint for: {display_name}")
    print(f"Audio file: {wav_path.name}")
    print(f"{'=' * 60}")
    
    try:
        # Get duration of audio file
        try:
            audio_duration = librosa.get_duration(path=str(wav_path))
            print(f"  Audio duration: {audio_duration:.2f} seconds")
        except Exception as e:
            print(f"  Warning: Could not get audio duration: {e}")
            audio_duration = 0.0
        
        # Extract XTTS embeddings
        print(f"  Extracting XTTS embeddings...")
        with torch.no_grad():
            gpt_cond_latent, speaker_embedding = xtts_wrapper.xtts_model.get_conditioning_latents(
                str(wav_path), 16000
            )
        
        print(f"  ✓ XTTS shapes - GPT: {gpt_cond_latent.shape}, Speaker: {speaker_embedding.shape}")
        
        # Convert XTTS tensors to Python lists for DuckDB
        gpt_latent_flat = gpt_cond_latent.cpu().numpy().flatten().tolist()
        xtts_embedding_flat = speaker_embedding.cpu().numpy().flatten().tolist()
        
        # Convert XTTS shapes to JSON strings
        gpt_shape_json = json.dumps(list(gpt_cond_latent.shape))
        xtts_shape_json = json.dumps(list(speaker_embedding.shape))
        
        # Extract ECAPA embedding
        print(f"  Extracting ECAPA embedding...")
        ecapa_embedding = await asyncio.to_thread(
            ecapa_processor.extract_embedding_from_file,
            wav_path,
            ecapa_processor.sample_rate
        )
        
        if ecapa_embedding is None:
            print(f"  Warning: Failed to extract ECAPA embedding, storing XTTS data only")
            ecapa_embedding_flat = None
        else:
            ecapa_embedding_flat = ecapa_embedding.flatten().tolist()
            print(f"  ✓ ECAPA embedding shape: {ecapa_embedding.shape}")
        
        # Store in database
        def insert_speaker_data():
            con.execute("""
                INSERT INTO speakers 
                (firstname, surname, gpt_cond_latent, gpt_shape, xtts_embedding, xtts_shape, 
                ecapa_embedding, total_duration_sec, sample_count, last_updated) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                firstname,
                surname,
                gpt_latent_flat,
                gpt_shape_json,
                xtts_embedding_flat,
                xtts_shape_json,
                ecapa_embedding_flat,
                audio_duration,
                1,
            ))
        
        await asyncio.to_thread(insert_speaker_data)
        
        print(f"  ✓ Successfully stored speaker imprint")
        print(f"  Duration: {audio_duration:.2f}s, Sample count: 1")
        return True
        
    except Exception as e:
        print(f"  ✗ Error creating speaker imprint: {e}")
        return False


async def populate_speakers():
    """Main function to populate speakers table"""
    print("=" * 80)
    print("SPEAKER POPULATION SCRIPT")
    print("=" * 80)
    print(f"Device: {CONFIG['device']}")
    print(f"Database: {CONFIG['duckdb_path']}\n")
    
    # Connect to database
    con = duckdb.connect(CONFIG["duckdb_path"])
    
    try:
        # Check if speakers table exists
        table_exists = con.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = 'speakers'
        """).fetchone()[0] > 0
        
        if not table_exists:
            print("❌ ERROR: Speakers table does not exist!")
            print("   Please run setup_database.py first to create the table structure.")
            return
        
        # Check current speaker count
        current_count = con.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]
        print(f"Current speakers in database: {current_count}\n")
        
        if current_count > 0:
            response = input("⚠ Database already contains speakers. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
            print()
        
        # Load models
        print("Loading AI models (this may take a few minutes)...\n")
        xtts_wrapper = SimpleXTTSWrapper(CONFIG["xtts_model_dir"], CONFIG["device"])
        ecapa_processor = SimpleECAPAProcessor(
            CONFIG["ecapa_tdnn_model_path"],
            CONFIG["device"],
            CONFIG["sample_rate"]
        )
        
        print(f"\n{'=' * 80}")
        print(f"Adding {len(INITIAL_SPEAKERS)} speakers...")
        print(f"{'=' * 80}\n")
        
        # Create speaker imprints
        results = await asyncio.gather(*[
            create_initial_speaker_imprint(
                con,
                xtts_wrapper,
                ecapa_processor,
                speaker["wav_path"],
                speaker["firstname"],
                speaker.get("surname")
            )
            for speaker in INITIAL_SPEAKERS
        ])
        
        # Summary
        successful = sum(results)
        failed = len(results) - successful
        
        print(f"\n{'=' * 80}")
        print("INITIAL SPEAKER CREATION - SUMMARY")
        print(f"{'=' * 80}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        # Perform sequential updates if configured
        if SPEAKER_UPDATES and successful > 0:
            print(f"\n{'=' * 80}")
            print(f"SEQUENTIAL UPDATES - Adding more samples to improve embeddings")
            print(f"{'=' * 80}\n")
            
            for update_config in SPEAKER_UPDATES:
                firstname = update_config["firstname"]
                surname = update_config.get("surname")
                wav_paths = update_config["wav_paths"]
                
                display_name = f"{firstname} {surname if surname else ''}"
                
                # Find speaker UID
                if surname:
                    uid_result = con.execute("""
                        SELECT uid FROM speakers 
                        WHERE firstname = ? AND surname = ?
                    """, (firstname, surname)).fetchone()
                else:
                    uid_result = con.execute("""
                        SELECT uid FROM speakers 
                        WHERE firstname = ? AND surname IS NULL
                    """, (firstname,)).fetchone()
                
                if not uid_result:
                    print(f"⚠ Speaker '{display_name}' not found, skipping updates")
                    continue
                
                speaker_uid = uid_result[0]
                
                print(f"Updating {display_name} (UID: {speaker_uid})")
                print(f"  Adding {len(wav_paths)} samples...")
                
                # Get initial state
                initial_state = con.execute("""
                    SELECT total_duration_sec, sample_count 
                    FROM speakers 
                    WHERE uid = ?
                """, (speaker_uid,)).fetchone()
                initial_duration, initial_count = initial_state
                
                # Process updates sequentially
                successful_updates = 0
                for i, wav_path in enumerate(wav_paths, 1):
                    wav_name = Path(wav_path).name
                    print(f"  [{i}/{len(wav_paths)}] {wav_name}...", end=" ")
                    
                    try:
                        success = await asyncio.to_thread(
                            ecapa_processor.update_speaker_imprint_from_file,
                            con,
                            wav_path,
                            speaker_uid
                        )
                        if success:
                            successful_updates += 1
                            print("✓")
                        else:
                            print("✗ Failed")
                    except FileNotFoundError:
                        print("✗ File not found")
                    except Exception as e:
                        print(f"✗ Error: {e}")
                
                # Get final state
                final_state = con.execute("""
                    SELECT total_duration_sec, sample_count 
                    FROM speakers 
                    WHERE uid = ?
                """, (speaker_uid,)).fetchone()
                final_duration, final_count = final_state
                
                print(f"  Results: {successful_updates}/{len(wav_paths)} successful")
                print(f"  Duration: {initial_duration:.2f}s → {final_duration:.2f}s (+{final_duration - initial_duration:.2f}s)")
                print(f"  Samples: {initial_count} → {final_count} (+{final_count - initial_count})")
                print()
        
        # Show final speaker list
        final_speakers = con.execute("""
            SELECT uid, firstname, surname, total_duration_sec, sample_count
            FROM speakers
            ORDER BY uid
        """).fetchall()
        
        print(f"{'=' * 80}")
        print("FINAL SPEAKER LIST")
        print(f"{'=' * 80}")
        print(f"Total speakers: {len(final_speakers)}\n")
        
        for row in final_speakers:
            uid, firstname, surname, duration, count = row
            name = f"{firstname} {surname if surname else ''}"
            print(f"  Speaker #{uid}: {name}")
            print(f"    Duration: {duration:.2f}s")
            print(f"    Samples: {count}")
            print()
        
        print(f"{'=' * 80}")
        print("✓ SPEAKER POPULATION COMPLETE!")
        print(f"{'=' * 80}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        con.close()


if __name__ == "__main__":
    print("\n⚠ WARNING: This script loads heavy AI models.")
    print("   Estimated time: 2-5 minutes")
    print("   GPU required for best performance\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(populate_speakers())
    else:
        print("Aborted.")
