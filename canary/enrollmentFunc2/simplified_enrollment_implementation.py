"""
SIMPLIFIED ENROLLMENT RECORDING IMPLEMENTATION
Integrates directly with existing process_audio_from_queue() loop

This approach:
1. Uses existing infrastructure (no new helper functions needed)
2. In-memory buffering (write WAV only on completion)
3. Natural integration with existing audio processing
4. Separate notification to Rasa on completion
"""

import random
from difflib import SequenceMatcher
import re
from pathlib import Path
import numpy as np
import wave
from typing import Optional, Tuple, Dict, Any, List
import asyncio


# =============================================================================
# PART 1: SIMPLE RECORD_PANGRAM (JUST KICKS OFF RECORDING)
# =============================================================================

async def record_pangram(
    client_id: str,
    uid: Optional[int] = None,
    firstname: str = None,
    surname: str = None
) -> Dict[str, Any]:
    """
    Simplified record_pangram - just selects pangram, sends to client, and sets flag.
    The actual recording happens in process_audio_from_queue() loop.
    
    Args:
        client_id: Unique session identifier
        uid: Speaker UID if updating existing speaker, None for new speaker
        firstname: First name of speaker
        surname: Surname of speaker
        
    Returns:
        Dict with status 'started' and pangram info
    """
    
    try:
        # 1. Select appropriate pangram
        pangram_id, pangram_text = await select_pangram_for_speaker(uid)
        
        if pangram_id is None:
            return {
                'status': 'error',
                'message': 'Failed to select pangram'
            }
        
        # 2. Initialize enrollment state in client queue
        if client_id not in client_queues:
            return {
                'status': 'error',
                'message': 'Client not found'
            }
        
        client_queues[client_id]["enrollment_state"] = {
            "recording_active": True,
            "pangram_id": pangram_id,
            "pangram_text": pangram_text,
            "uid": uid,
            "firstname": firstname,
            "surname": surname,
            "audio_buffer": [],  # List of audio chunks (int16 arrays)
            "transcript_buffer": [],  # List of on-topic utterances
            "off_topic_count": 0,
            "last_speech_time": asyncio.get_event_loop().time(),
            "start_time": asyncio.get_event_loop().time(),
            "last_prompt_time": None
        }
        
        # 3. Send pangram text to client (no TTS, just text)
        message_json = {
            "type": "enrollment_prompt",
            "text": pangram_text
        }
        await send_message_to_client(client_id, json.dumps(message_json))
        
        print(f"[Enrollment] Started recording for client {client_id}, pangram {pangram_id}")
        
        return {
            'status': 'started',
            'pangram_id': pangram_id,
            'message': 'Enrollment recording started'
        }
        
    except Exception as e:
        print(f"[Enrollment] Error starting recording: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }


# =============================================================================
# PART 2: ENROLLMENT HANDLER (CALLED FROM PROCESS_AUDIO_FROM_QUEUE LOOP)
# =============================================================================

async def handle_enrollment_recording(
    client_id: str,
    audio_chunk: np.ndarray,  # Current audio chunk (int16)
    is_voice_active: bool,
    current_transcript: str,  # Current utterance transcript (if any)
    speaker_name: str,
    speaker_uid: Optional[int],
    speaker_confidence: str,
    ecapa_processor: 'ECAPASpeakerProcessor'
) -> Optional[str]:
    """
    Handle enrollment recording logic within the main audio processing loop.
    
    This function is called from process_audio_from_queue() when recording_active is True.
    
    Args:
        client_id: Session ID
        audio_chunk: Current audio chunk being processed
        is_voice_active: Whether VAD detected voice in current chunk
        current_transcript: Final transcript of current utterance (empty string if none)
        speaker_name: ECAPA identified speaker name
        speaker_uid: ECAPA identified speaker UID
        speaker_confidence: ECAPA confidence level
        ecapa_processor: ECAPA processor instance
        
    Returns:
        Status string: 'success', 'aborted', or None (continue recording)
    """
    
    enrollment_state = client_queues[client_id]["enrollment_state"]
    current_time = asyncio.get_event_loop().time()
    
    # Constants
    SILENCE_TIMEOUT = 7.0  # seconds
    PROMPT_REMINDER_INTERVAL = 2.0  # seconds
    MAX_OFF_TOPIC_UTTERANCES = 3
    MIN_MATCH_THRESHOLD = 0.90  # 90% match to complete
    
    # =========================================================================
    # CHECK 1: ABORT IF OTHER SPEAKER DETECTED (HIGH CONFIDENCE)
    # =========================================================================
    if is_voice_active and speaker_confidence == "certain":
        expected_uid = enrollment_state["uid"]
        
        # If enrolling new speaker (uid=None) - ANY positive match is a problem
        if expected_uid is None and speaker_uid is not None:
            print(f"[Enrollment] ABORT: Other speaker detected during new enrollment (UID: {speaker_uid})")
            return await abort_enrollment(
                client_id,
                "Aborting imprint, please try again later with no other speakers present.",
                use_tts=True
            )
        
        # If updating existing speaker - different UID detected
        elif expected_uid is not None and speaker_uid != expected_uid:
            print(f"[Enrollment] ABORT: Different speaker detected (expected UID {expected_uid}, got {speaker_uid})")
            return await abort_enrollment(
                client_id,
                "Aborting imprint, please try again later with no other speakers present.",
                use_tts=True
            )
    
    # =========================================================================
    # PROCESS AUDIO: ADD TO BUFFER IF VOICE ACTIVE
    # =========================================================================
    if is_voice_active:
        enrollment_state["audio_buffer"].append(audio_chunk)
        enrollment_state["last_speech_time"] = current_time
    
    # =========================================================================
    # CHECK 2: PROCESS NEW TRANSCRIPT (FINAL UTTERANCE)
    # =========================================================================
    if current_transcript:  # New final utterance available
        print(f"[Enrollment] New utterance: '{current_transcript}'")
        
        # Check for cancel command
        if is_cancel_command(current_transcript):
            print(f"[Enrollment] ABORT: User cancelled")
            return await abort_enrollment(
                client_id,
                "Aborting imprint, please try again later.",
                use_tts=True
            )
        
        # Check if utterance is on-topic
        pangram_text = enrollment_state["pangram_text"]
        
        if is_utterance_on_topic(current_transcript, pangram_text):
            # On-topic: add to transcript buffer
            enrollment_state["transcript_buffer"].append(current_transcript)
            print(f"[Enrollment] On-topic utterance added to buffer")
            
            # Check if pangram is complete
            combined_transcript = " ".join(enrollment_state["transcript_buffer"])
            match_score = calculate_fuzzy_match(combined_transcript, pangram_text)
            
            print(f"[Enrollment] Match score: {match_score:.2%}")
            
            if match_score >= MIN_MATCH_THRESHOLD:
                print(f"[Enrollment] SUCCESS: Pangram completed (match: {match_score:.2%})")
                return await complete_enrollment(client_id, ecapa_processor)
        
        else:
            # Off-topic: increment counter
            enrollment_state["off_topic_count"] += 1
            print(f"[Enrollment] Off-topic utterance (count: {enrollment_state['off_topic_count']})")
            
            if enrollment_state["off_topic_count"] >= MAX_OFF_TOPIC_UTTERANCES:
                print(f"[Enrollment] ABORT: Too many off-topic utterances")
                return await abort_enrollment(
                    client_id,
                    "Aborting imprint, please try again later.",
                    use_tts=True
                )
    
    # =========================================================================
    # CHECK 3: SILENCE TIMEOUT
    # =========================================================================
    silence_duration = current_time - enrollment_state["last_speech_time"]
    
    if silence_duration >= SILENCE_TIMEOUT:
        print(f"[Enrollment] ABORT: Silence timeout ({silence_duration:.1f}s)")
        return await abort_enrollment(
            client_id,
            "Aborting imprint, please try again later.",
            use_tts=True
        )
    
    # =========================================================================
    # CHECK 4: PROMPT USER TO CONTINUE (IF PAUSED)
    # =========================================================================
    if silence_duration >= PROMPT_REMINDER_INTERVAL:
        last_prompt = enrollment_state["last_prompt_time"]
        
        # Only prompt if we haven't prompted recently
        if last_prompt is None or (current_time - last_prompt) >= PROMPT_REMINDER_INTERVAL:
            # Only prompt if user has started but not finished
            if len(enrollment_state["transcript_buffer"]) > 0:
                # Send text-only reminder (no TTS to avoid interrupting)
                message_json = {
                    "type": "enrollment_reminder",
                    "text": "Please finish reciting the prompt!"
                }
                await send_message_to_client(client_id, json.dumps(message_json))
                enrollment_state["last_prompt_time"] = current_time
    
    # Continue recording
    return None


# =============================================================================
# COMPLETION/ABORT HANDLERS
# =============================================================================

async def complete_enrollment(
    client_id: str,
    ecapa_processor: 'ECAPASpeakerProcessor'
) -> str:
    """
    Complete enrollment successfully: save audio, update imprint, notify Rasa.
    
    Returns:
        'success' status
    """
    enrollment_state = client_queues[client_id]["enrollment_state"]
    
    try:
        # 1. Save audio buffer to WAV file
        wav_path = await save_audio_buffer_to_wav(
            client_id,
            enrollment_state["audio_buffer"],
            enrollment_state["pangram_id"],
            enrollment_state["uid"]
        )
        
        print(f"[Enrollment] Saved audio to: {wav_path}")
        
        # 2. Create or update speaker imprint
        uid = enrollment_state["uid"]
        firstname = enrollment_state["firstname"]
        surname = enrollment_state["surname"]
        
        if uid is None:
            # Create new speaker
            success = await ecapa_processor.create_initial_speaker_imprint(
                wav_path=str(wav_path),
                firstname=firstname,
                surname=surname
            )
            
            if success:
                # Get newly created UID
                new_speaker = con.execute("""
                    SELECT uid FROM speakers 
                    WHERE firstname = ? AND surname = ?
                    ORDER BY uid DESC LIMIT 1
                """, [firstname, surname]).fetchone()
                
                if new_speaker:
                    new_uid = new_speaker[0]
                    await mark_pangram_as_recited(new_uid, enrollment_state["pangram_id"])
                    print(f"[Enrollment] Created new speaker with UID {new_uid}")
        else:
            # Update existing speaker
            success = await ecapa_processor.update_speaker_imprint_from_file(
                wav_path=str(wav_path),
                uid=uid
            )
            
            if success:
                await mark_pangram_as_recited(uid, enrollment_state["pangram_id"])
                print(f"[Enrollment] Updated speaker UID {uid}")
        
        # 3. Send success message to client
        success_message = "Enrollment completed successfully!"
        await send_message_to_client(
            client_id,
            json.dumps({"type": "enrollment_complete", "text": success_message})
        )
        if not clientSideTTS:
            asyncio.create_task(stream_tts_audio(client_id, success_message))
        
        # 4. Notify Rasa of completion
        await notify_rasa_enrollment_complete(client_id, 'success')
        
        # 5. Cleanup
        del client_queues[client_id]["enrollment_state"]
        
        return 'success'
        
    except Exception as e:
        print(f"[Enrollment] Error completing enrollment: {e}")
        return await abort_enrollment(client_id, "Enrollment failed due to an error.", use_tts=True)


async def abort_enrollment(
    client_id: str,
    message: str,
    use_tts: bool = True
) -> str:
    """
    Abort enrollment: save audio for debugging, notify Rasa, cleanup.
    
    Returns:
        'aborted' status
    """
    enrollment_state = client_queues[client_id]["enrollment_state"]
    
    try:
        # 1. Save audio buffer to WAV file (for debugging)
        wav_path = await save_audio_buffer_to_wav(
            client_id,
            enrollment_state["audio_buffer"],
            enrollment_state["pangram_id"],
            enrollment_state["uid"]
        )
        
        print(f"[Enrollment] Aborted, audio saved to: {wav_path}")
        
        # 2. Send abort message to client
        await send_message_to_client(
            client_id,
            json.dumps({"type": "enrollment_aborted", "text": message})
        )
        if use_tts and not clientSideTTS:
            asyncio.create_task(stream_tts_audio(client_id, message))
        
        # 3. Notify Rasa of abort
        await notify_rasa_enrollment_complete(client_id, 'aborted')
        
        # 4. Cleanup
        del client_queues[client_id]["enrollment_state"]
        
        return 'aborted'
        
    except Exception as e:
        print(f"[Enrollment] Error during abort: {e}")
        # Cleanup anyway
        if "enrollment_state" in client_queues[client_id]:
            del client_queues[client_id]["enrollment_state"]
        return 'aborted'


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def save_audio_buffer_to_wav(
    client_id: str,
    audio_buffer: List[np.ndarray],
    pangram_id: int,
    uid: Optional[int]
) -> Path:
    """
    Save accumulated audio buffer to WAV file.
    
    Args:
        client_id: Session ID
        audio_buffer: List of int16 audio chunks
        pangram_id: ID of pangram being recorded
        uid: Speaker UID (None for new speaker)
        
    Returns:
        Path to saved WAV file
    """
    # Create filename
    session_id = client_id.replace('-', '')[:8]
    
    if uid is not None:
        filename = f"pangram{pangram_id}_{session_id}_uid{uid}.wav"
    else:
        filename = f"pangram{pangram_id}_{session_id}.wav"
    
    wav_path = Path(CONFIG['samples_path']) / filename
    
    # Concatenate all audio chunks
    if not audio_buffer:
        print("[Enrollment] Warning: Empty audio buffer")
        concatenated = np.array([], dtype=np.int16)
    else:
        concatenated = np.concatenate(audio_buffer)
    
    # Write to WAV file using wave module (16-bit, 16kHz, mono)
    with wave.open(str(wav_path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(16000)  # 16kHz
        wav_file.writeframes(concatenated.tobytes())
    
    print(f"[Enrollment] Saved {len(concatenated)} samples to {wav_path}")
    
    return wav_path


async def notify_rasa_enrollment_complete(client_id: str, status: str):
    """
    Notify Rasa that enrollment is complete.
    
    This sends a message back through the normal conversation flow
    that Rasa will process to reset enrollment_active slot.
    
    Args:
        client_id: Session ID
        status: 'success' or 'aborted'
    """
    try:
        # Send special system message to Rasa
        message_for_rasa = {
            "sender": client_id,
            "message": f"SYSTEM_ENROLLMENT_COMPLETE_{status.upper()}",
            "metadata": {}
        }
        
        # Post to Rasa webhook
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{CONFIG['rasa_url']}/webhooks/rest/webhook",
                json=message_for_rasa,
                timeout=5
            )
        
        print(f"[Enrollment] Notified Rasa of completion: {status}")
        
    except Exception as e:
        print(f"[Enrollment] Error notifying Rasa: {e}")


async def select_pangram_for_speaker(uid: Optional[int]) -> Tuple[Optional[int], Optional[str]]:
    """
    Select a pangram for the speaker to recite.
    Prioritizes pangrams they haven't recited yet.
    
    Args:
        uid: Speaker UID, or None for new speaker
        
    Returns:
        Tuple of (pangram_id, pangram_text) or (None, None) if error
    """
    try:
        # Get all available pangrams
        all_pangrams = con.execute("""
            SELECT id, text FROM pangrams ORDER BY id
        """).fetchall()
        
        if not all_pangrams:
            print("[Enrollment] Error: No pangrams found in database")
            return None, None
        
        if uid is None:
            # New speaker - select random pangram
            selected = random.choice(all_pangrams)
            return selected[0], selected[1]
        
        # Get pangrams already recited by this speaker
        recited = con.execute("""
            SELECT pangrams FROM speakers WHERE uid = ?
        """, [uid]).fetchone()
        
        if recited is None:
            print(f"[Enrollment] Warning: Speaker UID {uid} not found")
            selected = random.choice(all_pangrams)
            return selected[0], selected[1]
        
        recited_ids = recited[0] if recited[0] else []
        
        # Filter out already-recited pangrams
        available_pangrams = [
            p for p in all_pangrams 
            if p[0] not in recited_ids
        ]
        
        if available_pangrams:
            selected = random.choice(available_pangrams)
        else:
            # All pangrams recited - select any
            selected = random.choice(all_pangrams)
        
        print(f"[Enrollment] Selected pangram {selected[0]} for speaker UID {uid}")
        return selected[0], selected[1]
        
    except Exception as e:
        print(f"[Enrollment] Error selecting pangram: {e}")
        return None, None


async def mark_pangram_as_recited(uid: int, pangram_id: int):
    """Mark a pangram as recited by adding it to speaker's pangrams array"""
    try:
        result = con.execute("""
            SELECT pangrams FROM speakers WHERE uid = ?
        """, [uid]).fetchone()
        
        if result is None:
            print(f"[Enrollment] Warning: Speaker UID {uid} not found")
            return
        
        current_pangrams = result[0] if result[0] else []
        
        if pangram_id not in current_pangrams:
            current_pangrams.append(pangram_id)
            
            con.execute("""
                UPDATE speakers 
                SET pangrams = ? 
                WHERE uid = ?
            """, [current_pangrams, uid])
            
            print(f"[Enrollment] Marked pangram {pangram_id} as recited for UID {uid}")
    
    except Exception as e:
        print(f"[Enrollment] Error marking pangram: {e}")


def normalize_text_for_comparison(text: str) -> str:
    """Normalize text for fuzzy matching"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    return text.strip()


def calculate_fuzzy_match(spoken_text: str, target_text: str) -> float:
    """
    Calculate fuzzy match score between spoken and target text.
    
    Returns:
        Float between 0.0 and 1.0
    """
    spoken_normalized = normalize_text_for_comparison(spoken_text)
    target_normalized = normalize_text_for_comparison(target_text)
    
    matcher = SequenceMatcher(None, spoken_normalized, target_normalized)
    return matcher.ratio()


def is_utterance_on_topic(utterance: str, pangram_text: str) -> bool:
    """
    Check if utterance contains words from the pangram.
    
    Returns:
        True if at least 30% of utterance words are in pangram
    """
    utterance_normalized = normalize_text_for_comparison(utterance)
    pangram_normalized = normalize_text_for_comparison(pangram_text)
    
    utterance_words = set(utterance_normalized.split())
    pangram_words = set(pangram_normalized.split())
    
    if not utterance_words:
        return False
    
    overlap = utterance_words & pangram_words
    overlap_ratio = len(overlap) / len(utterance_words)
    
    return overlap_ratio >= 0.3


def is_cancel_command(transcript: str) -> bool:
    """Check if transcript contains cancel command"""
    normalized = transcript.lower().strip()
    return 'cancel' in normalized and 'imprint' in normalized


# =============================================================================
# PART 3: INTEGRATION INTO PROCESS_AUDIO_FROM_QUEUE
# =============================================================================

"""
INTEGRATION EXAMPLE - Add this to your process_audio_from_queue() function:

async def process_audio_from_queue(client_id, nemo_transcriber, nemo_vad, canary_qwen_transcriber):
    # ... existing code ...
    
    try:
        while client_id in client_queues:
            try:
                audio_data = await asyncio.wait_for(
                    client_queues[client_id]["incoming_audio"].get(), 
                    timeout=1.0
                )
                
                # ... existing VAD processing ...
                
                if is_voice_active_in_chunk:
                    # ... existing voice processing ...
                    
                    # NEW: CHECK FOR ENROLLMENT RECORDING
                    if "enrollment_state" in client_queues[client_id]:
                        if client_queues[client_id]["enrollment_state"]["recording_active"]:
                            enrollment_result = await handle_enrollment_recording(
                                client_id=client_id,
                                audio_chunk=audio_int16,  # Current audio chunk
                                is_voice_active=True,
                                current_transcript=FINAL_TRANSCRIPT_IF_AVAILABLE,  # Or "" if none
                                speaker_name=SPEAKER_NAME,
                                speaker_uid=SPEAKER_UID,
                                speaker_confidence=SPEAKER_CONFIDENCE,
                                ecapa_processor=ecapa_processor
                            )
                            
                            # Check if enrollment completed or aborted
                            if enrollment_result in ['success', 'aborted']:
                                print(f"[Enrollment] Completed with status: {enrollment_result}")
                                # Continue normal processing
                                continue
                
                else:  # VAD indicates silence
                    # ... existing silence processing ...
                    
                    # NEW: CHECK FOR ENROLLMENT RECORDING DURING SILENCE
                    if "enrollment_state" in client_queues[client_id]:
                        if client_queues[client_id]["enrollment_state"]["recording_active"]:
                            # Still call handler during silence for timeout checks
                            enrollment_result = await handle_enrollment_recording(
                                client_id=client_id,
                                audio_chunk=np.array([], dtype=np.int16),  # Empty chunk
                                is_voice_active=False,
                                current_transcript="",
                                speaker_name=SPEAKER_NAME,
                                speaker_uid=SPEAKER_UID,
                                speaker_confidence=SPEAKER_CONFIDENCE,
                                ecapa_processor=ecapa_processor
                            )
                            
                            if enrollment_result in ['success', 'aborted']:
                                print(f"[Enrollment] Completed with status: {enrollment_result}")
                                continue
                
                # ... rest of existing code ...
                
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error processing audio for {client_id}: {e}")
                break
    
    finally:
        print("Async Audio processing stopped")
"""
