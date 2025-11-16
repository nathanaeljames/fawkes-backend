"""
ENROLLMENT RECORDING - CORRECTED 3-CLASS IMPLEMENTATION
========================================================

CORRECTIONS APPLIED:
1. ✅ Removed check_timeout() - done directly in main loop
2. ✅ Removed send_reminder_if_needed() - done directly in main loop  
3. ✅ Made abort_recording() PUBLIC with reason-based approach
4. ✅ Simplified architecture - manager focuses on core logic only

"""

import asyncio
import json
import wave
import re
import numpy as np
import aiohttp
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


# =============================================================================
# CLASS 1: TEXT UTILITIES (Pure functions, no state)
# =============================================================================

class EnrollmentTextUtils:
    """
    Utility functions for text processing during enrollment.
    All methods are static - no instance state needed.
    
    Responsibilities:
    - Text normalization
    - Fuzzy matching
    - On-topic detection
    - Cancel command detection
    """
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text (lowercase, no punctuation, collapsed whitespace)
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)     # Collapse whitespace
        return text.strip()
    
    @staticmethod
    def calculate_fuzzy_match(spoken_text: str, target_text: str) -> float:
        """
        Calculate fuzzy match score between spoken and target text.
        
        Args:
            spoken_text: What the user said
            target_text: Expected pangram text
            
        Returns:
            Match score between 0.0 and 1.0
        """
        spoken_normalized = EnrollmentTextUtils.normalize_text(spoken_text)
        target_normalized = EnrollmentTextUtils.normalize_text(target_text)
        
        matcher = SequenceMatcher(None, spoken_normalized, target_normalized)
        return matcher.ratio()
    
    @staticmethod
    def is_utterance_on_topic(utterance: str, pangram_text: str, threshold: float = 0.3) -> bool:
        """
        Check if utterance contains words from the pangram.
        
        Args:
            utterance: User's utterance
            pangram_text: Expected pangram text
            threshold: Minimum overlap ratio (default 30%)
            
        Returns:
            True if at least threshold% of utterance words are in pangram
        """
        utterance_normalized = EnrollmentTextUtils.normalize_text(utterance)
        pangram_normalized = EnrollmentTextUtils.normalize_text(pangram_text)
        
        utterance_words = set(utterance_normalized.split())
        pangram_words = set(pangram_normalized.split())
        
        if not utterance_words:
            return False
        
        overlap = utterance_words & pangram_words
        overlap_ratio = len(overlap) / len(utterance_words)
        
        return overlap_ratio >= threshold
    
    @staticmethod
    def is_cancel_command(transcript: str) -> bool:
        """
        Check if transcript contains a cancel command.
        
        Args:
            transcript: User's transcript
            
        Returns:
            True if cancel command detected
        """
        normalized = transcript.lower().strip()
        cancel_patterns = [
            'cancel imprint',
            'cancel enrollment',
            'stop recording',
            'abort imprint',
            'stop imprint'
        ]
        return any(pattern in normalized for pattern in cancel_patterns)


# =============================================================================
# CLASS 2: ENROLLMENT RECORDING MANAGER (Core logic and state management)
# =============================================================================

class EnrollmentRecordingManager:
    """
    Manages voice enrollment recording sessions.
    
    Responsibilities:
    - Start/stop recording sessions
    - Process completed utterances
    - Complete or abort recordings
    - Save WAV files
    - Update database
    - Notify Rasa
    
    NOT responsible for:
    - Timeout checks (done in main loop)
    - Reminder sending (done in main loop)
    """
    
    def __init__(self, db_connection, config: Dict, client_queues: Dict):
        """
        Initialize the enrollment recording manager.
        
        Args:
            db_connection: Database connection
            config: Configuration dict with paths, URLs, etc.
            client_queues: Global client_queues dict
        """
        self.con = db_connection
        self.config = config
        self.client_queues = client_queues
        
        # Configuration
        self.min_match_threshold = config.get('enrollment_min_match', 0.90)
        self.max_off_topic_utterances = config.get('enrollment_max_off_topic', 3)

    # PUBLIC METHODS =========================================================================
    
    async def start_recording(
        self,
        client_id: str,
        uid: Optional[int],
        firstname: Optional[str],
        surname: Optional[str]
    ) -> Dict:
        """
        Start enrollment recording for a client.
        
        Args:
            client_id: The client's ID
            uid: Optional speaker UID (for updates)
            firstname: Optional first name
            surname: Optional surname
            
        Returns:
            Dict with status and details
        """
        if client_id not in self.client_queues:
            return {"status": "error", "message": "Client not connected"}
        
        # Check if already recording
        if "enrollment_state" in self.client_queues[client_id]:
            if self.client_queues[client_id]["enrollment_state"].get("recording_active"):
                return {"status": "error", "message": "Recording already active"}
        
        # Select pangram
        pangram_id, pangram_text = await self._select_pangram(uid)
        if pangram_id is None:
            return {"status": "error", "message": "No pangrams available"}
        
        # Initialize enrollment state
        current_time = asyncio.get_event_loop().time()
        self.client_queues[client_id]["enrollment_state"] = {
            "recording_active": True,
            "audio_buffer": [],  # List of numpy arrays
            "transcript_buffer": [],  # List of transcript strings
            "start_time": current_time,
            "last_prompt_time": None,
            "pangram_id": pangram_id,
            "pangram_text": pangram_text,
            "uid": uid,
            "firstname": firstname,
            "surname": surname,
            "off_topic_count": 0
        }
        
        print(f"[Enrollment] Recording started for {client_id}")
        print(f"[Enrollment] Pangram: {pangram_text}")
        
        # Send initial prompt
        prompt_text = f"Please read the following sentence: {pangram_text}"
        message = {
            "type": "enrollment_prompt",
            "message": prompt_text,
            "pangram": pangram_text
        }
        await send_message_to_client(client_id, json.dumps(message))
        
        # Optional TTS
        if not self.config.get('clientSideTTS', False):
            asyncio.create_task(stream_tts_audio(client_id, prompt_text))
        
        return {
            "status": "started",
            "pangram_id": pangram_id,
            "pangram_text": pangram_text
        }
    
    async def process_utterance(
        self,
        client_id: str,
        utterance_audio: bytes,
        utterance_transcript: str,
        speaker_match: str
    ) -> Optional[str]:
        """
        Process a completed utterance during enrollment recording.
        
        This is called from the main loop after silence threshold is reached
        and a full utterance is ready.
        
        Args:
            client_id: The client's ID
            utterance_audio: Complete audio for this utterance (bytes)
            utterance_transcript: Transcription of the utterance
            speaker_match: Speaker ID from ECAPA ("nomatch" or UID)
            
        Returns:
            'success', 'aborted', or None if still recording
        """
        if client_id not in self.client_queues:
            return 'aborted'
        
        if "enrollment_state" not in self.client_queues[client_id]:
            return None
        
        enrollment_state = self.client_queues[client_id]["enrollment_state"]
        
        if not enrollment_state["recording_active"]:
            return None
        
        print(f"[Enrollment] Processing utterance: '{utterance_transcript}'")
        
        # Convert audio bytes to numpy array and add to buffer
        audio_int16 = np.frombuffer(utterance_audio, dtype=np.int16)
        enrollment_state["audio_buffer"].append(audio_int16)
        
        # Check for cancellation
        if EnrollmentTextUtils.is_cancel_command(utterance_transcript):
            print(f"[Enrollment] Cancel keyword detected")
            return await self.abort_recording(client_id, reason="cancel")
        
        # Check for other speaker
        if speaker_match != "nomatch":
            print(f"[Enrollment] Other speaker detected: {speaker_match}")
            return await self.abort_recording(client_id, reason="other_speaker")
        
        # Check if transcript is on-topic
        pangram_text = enrollment_state["pangram_text"]
        
        if EnrollmentTextUtils.is_utterance_on_topic(utterance_transcript, pangram_text):
            # On-topic: add to transcript buffer
            enrollment_state["transcript_buffer"].append(utterance_transcript)
            print(f"[Enrollment] On-topic utterance added")
            
            # Check if pangram is complete
            combined_transcript = " ".join(enrollment_state["transcript_buffer"])
            match_score = EnrollmentTextUtils.calculate_fuzzy_match(
                combined_transcript,
                pangram_text
            )
            
            print(f"[Enrollment] Match score: {match_score:.2%}")
            
            if match_score >= self.min_match_threshold:
                print(f"[Enrollment] Pangram completed! (match: {match_score:.2%})")
                return await self._complete_recording(client_id)
        else:
            # Off-topic: increment counter
            enrollment_state["off_topic_count"] += 1
            print(f"[Enrollment] Off-topic utterance (count: {enrollment_state['off_topic_count']})")
            
            if enrollment_state["off_topic_count"] >= self.max_off_topic_utterances:
                print(f"[Enrollment] Too many off-topic utterances")
                return await self.abort_recording(client_id, reason="off_topic")
        
        return None  # Still recording
    
    async def abort_recording(
        self,
        client_id: str,
        reason: str = "other_speaker"
    ) -> str:
        """
        Abort recording (can be called publicly).
        
        Args:
            client_id: Session ID
            reason: Reason for abort ('other_speaker', 'timeout', 'cancel', 'off_topic')
        
        Returns:
            'aborted'
        """
        messages = {
            "other_speaker": "Aborting imprint, please try again later with no other speakers present.",
            "timeout": "Aborting imprint, please try again later.",
            "off_topic": "Aborting imprint, please try again later.",
            "cancel": "Aborting imprint, please try again later."
        }
        
        message = messages.get(reason, "Aborting imprint, please try again later.")
        
        return await self._abort_recording(client_id, message, use_tts=True)
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    async def _complete_recording(self, client_id: str) -> str:
        """Complete the enrollment recording successfully."""
        if client_id not in self.client_queues:
            return 'aborted'
        
        if "enrollment_state" not in self.client_queues[client_id]:
            return 'aborted'
        
        enrollment_state = self.client_queues[client_id]["enrollment_state"]
        
        try:
            # Save WAV file
            wav_path = await self._save_audio_to_wav(client_id, enrollment_state)
            print(f"[Enrollment] Saved: {wav_path}")
            
            # Update speaker imprint (using ECAPA)
            # NOTE: You'll need to call your ECAPA processor here
            # ecapa_processor.update_speaker_imprint(uid, wav_path)
            
            # Mark pangram as recited
            if enrollment_state["uid"] is not None:
                await self._mark_pangram_recited(
                    enrollment_state["uid"],
                    enrollment_state["pangram_id"]
                )
            
            # Notify user
            success_text = "Enrollment completed successfully!"
            message = {"type": "enrollment_complete", "text": success_text}
            await send_message_to_client(client_id, json.dumps(message))
            
            if not self.config.get('clientSideTTS', False):
                asyncio.create_task(stream_tts_audio(client_id, success_text))
            
            # Notify Rasa
            await self._notify_rasa(client_id, 'success')
            
            # Cleanup
            enrollment_state["recording_active"] = False
            del self.client_queues[client_id]["enrollment_state"]
            
            return 'success'
            
        except Exception as e:
            print(f"[Enrollment] Error completing: {e}")
            return await self._abort_recording(client_id, "Enrollment failed.", use_tts=True)
    
    async def _abort_recording(self, client_id: str, message: str, use_tts: bool = True) -> str:
        """Internal abort method (called by public abort_recording)."""
        if client_id not in self.client_queues:
            return 'aborted'
        
        if "enrollment_state" not in self.client_queues[client_id]:
            return 'aborted'
        
        enrollment_state = self.client_queues[client_id]["enrollment_state"]
        
        try:
            # Save WAV for debugging (if any audio was captured)
            if enrollment_state["audio_buffer"]:
                wav_path = await self._save_audio_to_wav(client_id, enrollment_state)
                print(f"[Enrollment] Aborted, saved debug file: {wav_path}")
            
            # Notify user
            abort_message = {"type": "enrollment_aborted", "text": message}
            await send_message_to_client(client_id, json.dumps(abort_message))
            
            if use_tts and not self.config.get('clientSideTTS', False):
                asyncio.create_task(stream_tts_audio(client_id, message))
            
            # Notify Rasa
            await self._notify_rasa(client_id, 'aborted')
            
            # Cleanup
            enrollment_state["recording_active"] = False
            del self.client_queues[client_id]["enrollment_state"]
            
            return 'aborted'
            
        except Exception as e:
            print(f"[Enrollment] Error during abort: {e}")
            if "enrollment_state" in self.client_queues[client_id]:
                del self.client_queues[client_id]["enrollment_state"]
            return 'aborted'
    
    async def _save_audio_to_wav(self, client_id: str, enrollment_state: Dict) -> Path:
        """Save audio buffer to WAV file."""
        session_id = client_id.replace('-', '')[:8]
        pangram_id = enrollment_state["pangram_id"]
        uid = enrollment_state["uid"]
        surname = enrollment_state.get("surname", "")
        firstname = enrollment_state.get("firstname", "")
        
        # Build filename with surname and firstname
        if uid is not None:
            if surname and firstname:
                filename = f"pangram{pangram_id}_{session_id}_{surname}_{firstname}_uid{uid}.wav"
            elif surname:
                filename = f"pangram{pangram_id}_{session_id}_{surname}_uid{uid}.wav"
            elif firstname:
                filename = f"pangram{pangram_id}_{session_id}_{firstname}_uid{uid}.wav"
            else:
                filename = f"pangram{pangram_id}_{session_id}_uid{uid}.wav"
        else:
            if surname and firstname:
                filename = f"pangram{pangram_id}_{session_id}_{surname}_{firstname}.wav"
            elif surname:
                filename = f"pangram{pangram_id}_{session_id}_{surname}.wav"
            elif firstname:
                filename = f"pangram{pangram_id}_{session_id}_{firstname}.wav"
            else:
                filename = f"pangram{pangram_id}_{session_id}.wav"
        
        wav_path = Path(self.config['samples_path']) / filename
        
        # Concatenate all audio chunks
        audio_buffer = enrollment_state["audio_buffer"]
        
        if not audio_buffer:
            concatenated = np.array([], dtype=np.int16)
        else:
            concatenated = np.concatenate(audio_buffer)
        
        # Save using wave module
        with wave.open(str(wav_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)
            wav_file.writeframes(concatenated.tobytes())
        
        return wav_path
    
    async def _mark_pangram_recited(self, uid: int, pangram_id: int):
        """Mark pangram as recited in database."""
        try:
            result = self.con.execute("""
                SELECT pangrams FROM speakers WHERE uid = ?
            """, [uid]).fetchone()
            
            if result is None:
                return
            
            current = result[0] if result[0] else []
            
            if pangram_id not in current:
                current.append(pangram_id)
                self.con.execute("""
                    UPDATE speakers SET pangrams = ? WHERE uid = ?
                """, [current, uid])
                print(f"[Enrollment] Marked pangram {pangram_id} for UID {uid}")
        
        except Exception as e:
            print(f"[Enrollment] Error marking pangram: {e}")
    
    async def _notify_rasa(self, client_id: str, status: str):
        """Notify Rasa of completion via trigger_intent."""
        try:
            intent_name = f"system_enrollment_complete_{status}"
            
            async with aiohttp.ClientSession() as session:
                response = await session.post(
                    f"{self.config['rasa_url']}/conversations/{client_id}/trigger_intent",
                    json={"name": intent_name, "entities": []},
                    params={"output_channel": "latest"},
                    timeout=aiohttp.ClientTimeout(total=5)
                )
                
                if response.status == 200:
                    print(f"[Enrollment] Triggered intent: {intent_name}")
                else:
                    print(f"[Enrollment] Failed to trigger intent: {response.status}")
                    
        except Exception as e:
            print(f"[Enrollment] Error notifying Rasa: {e}")
    
    async def _select_pangram(self, uid: Optional[int]) -> Tuple[Optional[int], Optional[str]]:
        """Select an appropriate pangram for the speaker."""
        try:
            if uid is not None:
                # Get pangrams already recited
                result = self.con.execute("""
                    SELECT pangrams FROM speakers WHERE uid = ?
                """, [uid]).fetchone()
                
                recited = result[0] if (result and result[0]) else []
                
                # Get unrecited pangrams
                placeholders = ','.join('?' * len(recited)) if recited else ''
                query = f"""
                    SELECT pangram_id, pangram_text FROM pangrams
                    WHERE pangram_id NOT IN ({placeholders})
                    ORDER BY RANDOM()
                    LIMIT 1
                """ if recited else """
                    SELECT pangram_id, pangram_text FROM pangrams
                    ORDER BY RANDOM()
                    LIMIT 1
                """
                
                result = self.con.execute(query, recited if recited else []).fetchone()
            else:
                # New speaker: any pangram
                result = self.con.execute("""
                    SELECT pangram_id, pangram_text FROM pangrams
                    ORDER BY RANDOM()
                    LIMIT 1
                """).fetchone()
            
            if result:
                return result[0], result[1]
            else:
                print("[Enrollment] No pangrams available")
                return None, None
                
        except Exception as e:
            print(f"[Enrollment] Error selecting pangram: {e}")
            return None, None


# =============================================================================
# CLASS 3: API HANDLER (API interface)
# =============================================================================

class EnrollmentAPIHandler:
    """
    Handles FastAPI endpoints for speaker enrollment workflows.
    
    This class should REPLACE or UPDATE your existing EnrollmentAPIHandler.
    
    Responsibilities:
    - Receive API requests
    - Validate input
    - Query database for speaker info
    - Delegate to EnrollmentRecordingManager
    - Return API responses
    """
    
    def __init__(self, db_connection, recording_manager: EnrollmentRecordingManager):
        """
        Initialize the API handler.
        
        Args:
            db_connection: Database connection
            recording_manager: EnrollmentRecordingManager instance
        """
        self.con = db_connection
        self.recording_manager = recording_manager
    
    async def query_speaker(self, request: dict) -> dict:
        """
        Query speaker information by UID or name.
        
        Args:
            request: Dict with 'uid' or 'firstname'/'surname'
            
        Returns:
            Dict with speaker info or error
        """
        try:
            uid = request.get("uid")
            firstname = request.get("firstname")
            surname = request.get("surname")
            
            if uid is not None:
                result = self.con.execute("""
                    SELECT uid, firstname, surname, pangrams 
                    FROM speakers WHERE uid = ?
                """, [uid]).fetchone()
            elif firstname and surname:
                result = self.con.execute("""
                    SELECT uid, firstname, surname, pangrams 
                    FROM speakers WHERE firstname = ? AND surname = ?
                """, [firstname, surname]).fetchone()
            else:
                return {"error": "Must provide uid or firstname+surname"}
            
            if result:
                return {
                    "found": True,
                    "uid": result[0],
                    "firstname": result[1],
                    "surname": result[2],
                    "pangrams_recited": result[3] if result[3] else []
                }
            else:
                return {"found": False}
                
        except Exception as e:
            return {"error": str(e)}
    
    async def record_pangram(self, request: dict) -> dict:
        """
        Start enrollment recording for a client.
        
        This is called from the Rasa action and returns immediately.
        
        Args:
            request: Dict with:
                - client_id: str (required)
                - uid: int (optional)
                - firstname: str (optional)
                - surname: str (optional)
                
        Returns:
            Dict with status and details
        """
        client_id = request.get("client_id")
        
        if not client_id:
            return {"status": "error", "message": "client_id required"}
        
        uid = request.get("uid")
        firstname = request.get("firstname")
        surname = request.get("surname")
        
        # Delegate to recording manager
        result = await self.recording_manager.start_recording(
            client_id=client_id,
            uid=uid,
            firstname=firstname,
            surname=surname
        )
        
        return result


# =============================================================================
# INTEGRATION WITH MAIN LOOP
# =============================================================================

"""
INTEGRATION INSTRUCTIONS (CORRECTED):

1. Initialize in main():
   
   enrollment_recording_manager = EnrollmentRecordingManager(
       db_connection=con,
       config=CONFIG,
       client_queues=client_queues
   )
   
   enrollment_api_handler = EnrollmentAPIHandler(
       db_connection=con,
       recording_manager=enrollment_recording_manager
   )

2. Update FastAPI endpoint:
   
   @app.post("/api/record_pangram")
   async def record_pangram_endpoint(request: dict):
       return await enrollment_api_handler.record_pangram(request)

3. In process_audio_from_queue(), add CLIENT-LEVEL last_utterance_time:
   
   async def process_audio_from_queue(client_id, ...):
       # ... existing setup ...
       
       # CLIENT-LEVEL timestamp (not in enrollment_state!)
       last_utterance_time = asyncio.get_event_loop().time()
       
       current_utterance_buffer = []
       is_speaking = False
       silence_counter = 0
       
       while True:
           audio_chunk = await client_queues[client_id]["incoming_audio"].get()
           current_time = asyncio.get_event_loop().time()
           
           # ... VAD processing ...
           
           # === VOICE ACTIVE SECTION ===
           if voice_active:
               if not is_speaking:
                   is_speaking = True
                   silence_counter = 0
               
               current_utterance_buffer.append(audio_chunk)
               # Just accumulate - no enrollment processing here
           
           # === SILENCE SECTION ===
           else:
               silence_counter += 1
               
               # === DIRECT TIMEOUT CHECK (no method call) ===
               if "enrollment_state" in client_queues[client_id]:
                   enrollment_state = client_queues[client_id]["enrollment_state"]
                   
                   if enrollment_state["recording_active"]:
                       silence_duration = current_time - last_utterance_time
                       
                       # Check timeout directly
                       if silence_duration >= CONFIG['enrollment_timeout']:
                           print(f"[Enrollment] Timeout after {silence_duration:.1f}s")
                           await enrollment_recording_manager.abort_recording(
                               client_id,
                               reason="timeout"
                           )
                       
                       # Send reminder directly (no method call)
                       elif silence_duration >= CONFIG['enrollment_reminder_interval']:
                           last_prompt_time = enrollment_state.get("last_prompt_time")
                           
                           if last_prompt_time is None or (current_time - last_prompt_time) >= CONFIG['enrollment_reminder_interval']:
                               pangram_text = enrollment_state["pangram_text"]
                               reminder_text = f"Still there? Please read: {pangram_text}"
                               
                               message = {
                                   "type": "enrollment_reminder",
                                   "message": reminder_text
                               }
                               await send_message_to_client(client_id, json.dumps(message))
                               
                               enrollment_state["last_prompt_time"] = current_time
                               print(f"[Enrollment] Reminder sent after {silence_duration:.1f}s")
               
               # Process utterance finality
               if is_speaking and silence_counter >= SILENCE_CHUNKS_THRESHOLD:
                   print("Acoustic finality detected")
                   
                   # UPDATE CLIENT-LEVEL last_utterance_time
                   last_utterance_time = current_time
                   
                   # Get final speaker match from ECAPA
                   final_speaker_match = "nomatch"
                   if len(current_utterance_buffer) > 0:
                       ecapa_result = await ecapa_processor.extract_and_match_from_buffer(
                           current_utterance_buffer,
                           reason="silence"
                       )
                       if "error" not in ecapa_result:
                           final_speaker_match = ecapa_result['speaker_result']
                   
                   # Get final transcript from Canary-Qwen
                   # (your existing transcription code)
                   final_transcription_text = ""  # ... your code ...
                   
                   # === PROCESS ENROLLMENT UTTERANCE ===
                   if "enrollment_state" in client_queues[client_id]:
                       if client_queues[client_id]["enrollment_state"]["recording_active"]:
                           # Combine utterance buffer into bytes
                           utterance_bytes = b''.join(current_utterance_buffer)
                           
                           result = await enrollment_recording_manager.process_utterance(
                               client_id=client_id,
                               utterance_audio=utterance_bytes,
                               utterance_transcript=final_transcription_text,
                               speaker_match=final_speaker_match
                           )
                           
                           if result in ['success', 'aborted']:
                               # Recording completed or aborted
                               pass
                   
                   # === SKIP RASA IF ENROLLMENT ACTIVE ===
                   if "enrollment_state" in client_queues[client_id]:
                       if client_queues[client_id]["enrollment_state"]["recording_active"]:
                           # Skip sending to Rasa during enrollment
                           pass
                       else:
                           # Normal Rasa processing
                           # ... your Rasa code ...
                   else:
                       # Normal Rasa processing
                       # ... your Rasa code ...
                   
                   # Reset for next utterance
                   current_utterance_buffer.clear()
                   is_speaking = False
"""


# =============================================================================
# CORRECTED IMPLEMENTATION CHECKLIST
# =============================================================================

"""
CORRECTED IMPLEMENTATION CHECKLIST:

✅ 1. THREE CLASSES with proper separation
   [ ] EnrollmentTextUtils (static text utilities)
   [ ] EnrollmentRecordingManager (core recording logic)
   [ ] EnrollmentAPIHandler (API interface)

✅ 2. PUBLIC abort_recording() with reason-based approach
   [ ] Takes reason parameter ('timeout', 'other_speaker', 'cancel', 'off_topic')
   [ ] Maps reasons to appropriate messages
   [ ] Can be called from anywhere (public method)

✅ 3. NO check_timeout() method
   [ ] Timeout check done DIRECTLY in main loop
   [ ] Simple comparison: silence_duration >= CONFIG['enrollment_timeout']

✅ 4. NO send_reminder_if_needed() method
   [ ] Reminder logic done DIRECTLY in main loop
   [ ] Check silence_duration and last_prompt_time directly

✅ 5. CLIENT-LEVEL last_utterance_time
   [ ] Initialized in process_audio_from_queue
   [ ] Updated when utterance completes
   [ ] Used for timeout and reminder checks

✅ 6. UTTERANCE-BY-UTTERANCE processing
   [ ] Voice active: just accumulate audio
   [ ] Silence: check timeout/reminders on every chunk
   [ ] After finality: process complete utterance

✅ 7. SIMPLIFIED architecture
   [ ] Manager focuses on core logic only
   [ ] Main loop handles timing checks directly
   [ ] Clear separation of concerns

WHAT WE REMOVED:
❌ check_timeout() method - now inline in main loop
❌ send_reminder_if_needed() method - now inline in main loop
❌ Private _abort_recording() as primary interface - now abort_recording() is public

WHAT WE FIXED:
✅ abort_recording() is public and takes reasons
✅ Timeout/reminder checks are direct (no method calls)
✅ Simpler, more maintainable architecture
"""
