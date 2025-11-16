"""
New record_pangram() function to add to server01e.py
This function handles the enrollment recording process with pangram prompts.

Add this function to the appropriate class (likely ECAPASpeakerProcessor or as a standalone async function)
"""

import random
import asyncio
from pathlib import Path
from difflib import SequenceMatcher
import re

async def record_pangram(
    client_id: str,
    uid: Optional[int] = None,
    firstname: str = None,
    surname: str = None,
    ecapa_processor: 'ECAPASpeakerProcessor' = None
) -> Dict[str, Any]:
    """
    Record a user reciting a pangram for speaker imprint creation/update.
    
    Args:
        client_id: Unique session identifier
        uid: Speaker UID if updating existing speaker, None for new speaker
        firstname: First name of speaker
        surname: Surname of speaker
        ecapa_processor: Reference to ECAPASpeakerProcessor instance
        
    Returns:
        Dict with status ('success' or 'aborted') and message
    """
    
    # === STEP 1: Initialize recording state ===
    recording_state = {
        'is_recording': False,
        'pangram_id': None,
        'pangram_text': None,
        'wav_file_path': None,
        'start_time': None,
        'utterances': [],  # List of transcribed utterances
        'silence_timer': None,
        'off_topic_count': 0,
        'last_utterance_time': None,
        'abort_reason': None
    }
    
    try:
        # === STEP 2: Select appropriate pangram ===
        pangram_id, pangram_text = await select_pangram_for_speaker(uid)
        
        if pangram_id is None:
            return {
                'status': 'aborted',
                'message': 'Failed to select pangram'
            }
        
        recording_state['pangram_id'] = pangram_id
        recording_state['pangram_text'] = pangram_text
        
        # === STEP 3: Set up WAV file path ===
        session_id = client_id.replace('-', '')[:8]  # Shortened session ID
        
        if uid is not None:
            filename = f"pangram{pangram_id}_{session_id}_uid{uid}.wav"
        else:
            filename = f"pangram{pangram_id}_{session_id}.wav"
        
        wav_path = Path(CONFIG['samples_path']) / filename
        recording_state['wav_file_path'] = wav_path
        
        # === STEP 4: Send pangram text to client (no TTS) ===
        await send_text_to_client(
            client_id,
            pangram_text,
            use_tts=False
        )
        
        # === STEP 5: Start recording ===
        recording_state['is_recording'] = True
        recording_state['start_time'] = asyncio.get_event_loop().time()
        recording_state['last_utterance_time'] = recording_state['start_time']
        
        # Initialize WAV file writer
        wav_writer = await initialize_wav_writer(
            wav_path,
            sample_rate=16000,
            channels=1,
            sample_width=2  # 16-bit
        )
        
        # === STEP 6: Main recording loop with ASR monitoring ===
        try:
            result = await recording_loop(
                client_id=client_id,
                recording_state=recording_state,
                wav_writer=wav_writer,
                uid=uid,
                ecapa_processor=ecapa_processor
            )
            
            if result['status'] == 'aborted':
                # Send abort message to user
                await send_text_and_audio_to_client(
                    client_id,
                    result['message']
                )
                return result
            
        finally:
            # Always close the WAV file
            await wav_writer.close()
            recording_state['is_recording'] = False
        
        # === STEP 7: Process completed recording ===
        if result['status'] == 'success':
            # Update speaker imprint
            if uid is None:
                # Create new speaker
                success = await ecapa_processor.create_initial_speaker_imprint(
                    wav_path=str(wav_path),
                    firstname=firstname,
                    surname=surname
                )
                
                if success:
                    # Get the newly created UID
                    new_speaker = con.execute("""
                        SELECT uid FROM speakers 
                        WHERE firstname = ? AND surname = ?
                        ORDER BY uid DESC LIMIT 1
                    """, [firstname, surname]).fetchone()
                    
                    if new_speaker:
                        new_uid = new_speaker[0]
                        # Add pangram to recited list
                        await mark_pangram_as_recited(new_uid, pangram_id)
                        
                        return {
                            'status': 'success',
                            'message': 'Speaker imprint created successfully',
                            'uid': new_uid
                        }
            else:
                # Update existing speaker
                success = await ecapa_processor.update_speaker_imprint_from_file(
                    wav_path=str(wav_path),
                    uid=uid
                )
                
                if success:
                    # Add pangram to recited list
                    await mark_pangram_as_recited(uid, pangram_id)
                    
                    return {
                        'status': 'success',
                        'message': 'Speaker imprint updated successfully',
                        'uid': uid
                    }
        
        return {
            'status': 'aborted',
            'message': 'Failed to process speaker imprint'
        }
        
    except Exception as e:
        print(f"[record_pangram] Error: {e}")
        return {
            'status': 'aborted',
            'message': f'Recording error: {str(e)}'
        }


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
            print("[record_pangram] Error: No pangrams found in database")
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
            print(f"[record_pangram] Warning: Speaker UID {uid} not found")
            # Still select a random pangram
            selected = random.choice(all_pangrams)
            return selected[0], selected[1]
        
        recited_ids = recited[0] if recited[0] else []
        
        # Filter out already-recited pangrams
        available_pangrams = [
            p for p in all_pangrams 
            if p[0] not in recited_ids
        ]
        
        if available_pangrams:
            # Select from un-recited pangrams
            selected = random.choice(available_pangrams)
        else:
            # All pangrams have been recited - select any random one
            selected = random.choice(all_pangrams)
        
        print(f"[record_pangram] Selected pangram {selected[0]} for speaker UID {uid}")
        return selected[0], selected[1]
        
    except Exception as e:
        print(f"[record_pangram] Error selecting pangram: {e}")
        return None, None


async def mark_pangram_as_recited(uid: int, pangram_id: int):
    """Mark a pangram as recited by adding it to the speaker's pangrams array"""
    try:
        # Get current pangrams list
        result = con.execute("""
            SELECT pangrams FROM speakers WHERE uid = ?
        """, [uid]).fetchone()
        
        if result is None:
            print(f"[record_pangram] Warning: Speaker UID {uid} not found")
            return
        
        current_pangrams = result[0] if result[0] else []
        
        # Add new pangram if not already present
        if pangram_id not in current_pangrams:
            current_pangrams.append(pangram_id)
            
            con.execute("""
                UPDATE speakers 
                SET pangrams = ? 
                WHERE uid = ?
            """, [current_pangrams, uid])
            
            print(f"[record_pangram] Marked pangram {pangram_id} as recited for speaker UID {uid}")
    
    except Exception as e:
        print(f"[record_pangram] Error marking pangram as recited: {e}")


async def recording_loop(
    client_id: str,
    recording_state: Dict,
    wav_writer: Any,
    uid: Optional[int],
    ecapa_processor: 'ECAPASpeakerProcessor'
) -> Dict[str, Any]:
    """
    Main recording loop that monitors ASR output and handles abort conditions.
    
    Returns:
        Dict with status ('success' or 'aborted') and message
    """
    
    pangram_text = recording_state['pangram_text']
    pangram_words = normalize_text_for_comparison(pangram_text)
    
    SILENCE_TIMEOUT = 7.0  # seconds
    PROMPT_REMINDER_INTERVAL = 2.0  # seconds
    MAX_OFF_TOPIC_UTTERANCES = 3
    
    last_prompt_time = None
    
    while recording_state['is_recording']:
        current_time = asyncio.get_event_loop().time()
        
        # Check for cancel command
        if await check_for_cancel_command(client_id):
            await send_text_and_audio_to_client(
                client_id,
                "Aborting imprint, please try again later."
            )
            return {
                'status': 'aborted',
                'message': 'User cancelled imprint'
            }
        
        # Check for other speaker detection
        if await check_for_other_speaker(client_id, uid, ecapa_processor):
            await send_text_and_audio_to_client(
                client_id,
                "Aborting imprint, please try again later with no other speakers present."
            )
            return {
                'status': 'aborted',
                'message': 'Other speaker detected'
            }
        
        # Get new utterances from ASR (final utterances, not interim)
        new_utterances = await get_new_utterances_for_client(client_id)
        
        if new_utterances:
            recording_state['last_utterance_time'] = current_time
            
            for utterance_text in new_utterances:
                recording_state['utterances'].append(utterance_text)
                
                # Check if utterance matches expected pangram content
                utterance_normalized = normalize_text_for_comparison(utterance_text)
                
                # Calculate fuzzy match score
                match_score = calculate_fuzzy_match(
                    ' '.join(recording_state['utterances']),
                    pangram_text
                )
                
                print(f"[record_pangram] Utterance match score: {match_score:.2f}")
                
                # Check if user has completed the pangram (90% match)
                if match_score >= 0.90:
                    print(f"[record_pangram] Pangram completed successfully (match: {match_score:.2f})")
                    return {
                        'status': 'success',
                        'message': 'Pangram recitation completed'
                    }
                
                # Check if utterance is off-topic
                if not is_utterance_on_topic(utterance_normalized, pangram_words):
                    recording_state['off_topic_count'] += 1
                    print(f"[record_pangram] Off-topic utterance detected (count: {recording_state['off_topic_count']})")
                    
                    if recording_state['off_topic_count'] >= MAX_OFF_TOPIC_UTTERANCES:
                        await send_text_and_audio_to_client(
                            client_id,
                            "Aborting imprint, please try again later."
                        )
                        return {
                            'status': 'aborted',
                            'message': 'Too many off-topic utterances'
                        }
        
        # Check for silence timeout
        silence_duration = current_time - recording_state['last_utterance_time']
        
        if silence_duration >= SILENCE_TIMEOUT:
            await send_text_and_audio_to_client(
                client_id,
                "Aborting imprint, please try again later."
            )
            return {
                'status': 'aborted',
                'message': 'Silence timeout exceeded'
            }
        
        # Send reminder if user paused during recitation
        if silence_duration >= PROMPT_REMINDER_INTERVAL:
            if last_prompt_time is None or (current_time - last_prompt_time) >= PROMPT_REMINDER_INTERVAL:
                # Check if user has started but not finished
                if len(recording_state['utterances']) > 0:
                    match_score = calculate_fuzzy_match(
                        ' '.join(recording_state['utterances']),
                        pangram_text
                    )
                    
                    if match_score < 0.90:  # Not completed yet
                        await send_text_to_client(
                            client_id,
                            "Please finish reciting the prompt!",
                            use_tts=False
                        )
                        last_prompt_time = current_time
        
        # Write audio data to WAV file (non-blocking)
        await write_audio_chunk_to_wav(client_id, wav_writer)
        
        # Small sleep to prevent busy-waiting
        await asyncio.sleep(0.05)
    
    return {
        'status': 'aborted',
        'message': 'Recording loop exited unexpectedly'
    }


def normalize_text_for_comparison(text: str) -> str:
    """Normalize text for fuzzy matching"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()


def calculate_fuzzy_match(spoken_text: str, target_text: str) -> float:
    """
    Calculate fuzzy match score between spoken and target text.
    
    Returns:
        Float between 0.0 and 1.0 indicating match quality
    """
    spoken_normalized = normalize_text_for_comparison(spoken_text)
    target_normalized = normalize_text_for_comparison(target_text)
    
    # Use SequenceMatcher for fuzzy matching
    matcher = SequenceMatcher(None, spoken_normalized, target_normalized)
    return matcher.ratio()


def is_utterance_on_topic(utterance: str, pangram_words: str) -> bool:
    """
    Check if an utterance contains words from the pangram.
    
    Args:
        utterance: Normalized utterance text
        pangram_words: Normalized pangram text
        
    Returns:
        True if utterance appears to be on-topic
    """
    utterance_words = set(utterance.split())
    pangram_word_set = set(pangram_words.split())
    
    # Check if at least 30% of utterance words are in pangram
    if not utterance_words:
        return False
    
    overlap = utterance_words & pangram_word_set
    overlap_ratio = len(overlap) / len(utterance_words)
    
    return overlap_ratio >= 0.3


async def check_for_cancel_command(client_id: str) -> bool:
    """Check if user said 'cancel imprint'"""
    # Get recent utterances
    utterances = await get_recent_utterances_for_client(client_id, max_count=3)
    
    for utterance in utterances:
        normalized = utterance.lower().strip()
        if 'cancel' in normalized and 'imprint' in normalized:
            return True
    
    return False


async def check_for_other_speaker(
    client_id: str,
    expected_uid: Optional[int],
    ecapa_processor: 'ECAPASpeakerProcessor'
) -> bool:
    """
    Check if ECAPA detected a different speaker during recording.
    
    Args:
        client_id: Session ID
        expected_uid: UID of speaker we're enrolling (None for new speaker)
        ecapa_processor: ECAPA processor instance
        
    Returns:
        True if another speaker was detected
    """
    # Get recent ECAPA identification results
    recent_matches = await get_recent_ecapa_matches_for_client(client_id, max_count=5)
    
    for match in recent_matches:
        matched_uid = match.get('uid')
        confidence = match.get('confidence', 0.0)
        
        # High confidence match
        if confidence > 0.7:
            if expected_uid is None:
                # We're enrolling a new speaker - any positive match is a problem
                return True
            elif matched_uid != expected_uid:
                # We're updating an existing speaker - different UID detected
                return True
    
    return False


# ===================================================================
# HELPER FUNCTIONS (these need to be implemented based on your system)
# ===================================================================

async def send_text_to_client(client_id: str, text: str, use_tts: bool = True):
    """Send text message to client (implement based on your WebSocket system)"""
    # TODO: Implement based on your existing send_message_to_client function
    pass


async def send_text_and_audio_to_client(client_id: str, text: str):
    """Send both text and TTS audio to client"""
    # TODO: Implement based on your existing TTS system
    pass


async def initialize_wav_writer(wav_path: Path, sample_rate: int, channels: int, sample_width: int):
    """Initialize WAV file writer (non-blocking)"""
    # TODO: Implement non-blocking WAV file writer
    # Return an object with a .close() method and ability to write chunks
    pass


async def write_audio_chunk_to_wav(client_id: str, wav_writer: Any):
    """Write audio chunk from client's incoming audio queue to WAV file"""
    # TODO: Implement based on your audio queue system
    pass


async def get_new_utterances_for_client(client_id: str) -> List[str]:
    """Get new final utterances (not interim results) for client"""
    # TODO: Implement based on your ASR system
    # Should return final Canary-Qwen transcriptions, not interim NeMo results
    pass


async def get_recent_utterances_for_client(client_id: str, max_count: int) -> List[str]:
    """Get recent utterances for checking cancel command"""
    # TODO: Implement
    pass


async def get_recent_ecapa_matches_for_client(client_id: str, max_count: int) -> List[Dict]:
    """Get recent ECAPA speaker identification results"""
    # TODO: Implement
    # Should return list of dicts with 'uid' and 'confidence' keys
    pass
