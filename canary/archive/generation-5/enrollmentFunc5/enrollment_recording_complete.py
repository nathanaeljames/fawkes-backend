"""
ENROLLMENT RECORDING - COMPLETE 3-CLASS IMPLEMENTATION
=======================================================

This file contains all three classes with full implementations:
1. EnrollmentTextUtils - Text processing utilities (static)
2. EnrollmentRecordingManager - Core recording logic and state management
3. AudioChunkProcessor - Helper for processing audio chunks (if needed)
4. TranscriptionProcessor - Helper for processing transcriptions (if needed)

KEY ARCHITECTURAL DECISIONS:
- last_utterance_time is CLIENT-LEVEL (not in enrollment_state)
- Process UTTERANCE-BY-UTTERANCE (not chunk-by-chunk)
- Main loop handles timeout checks and reminders
- EnrollmentRecordingManager handles recording logic

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
            True if transcript contains cancel keywords
        """
        cancel_keywords = [
            "cancel", "stop", "quit", "nevermind", "never mind",
            "abort", "forget it", "end", "exit"
        ]
        
        transcript_lower = transcript.lower()
        return any(keyword in transcript_lower for keyword in cancel_keywords)


# =============================================================================
# CLASS 2: ENROLLMENT RECORDING MANAGER (Core logic and state management)
# =============================================================================

class EnrollmentRecordingManager:
    """
    Manages voice enrollment recording sessions.
    
    Responsibilities:
    - Start/stop recording sessions
    - Process completed utterances
    - Check timeout and reminders
    - Complete or abort recordings
    - Save WAV files
    - Update database
    - Notify Rasa
    """
    
    def __init__(self, db_connection, config: Dict, client_queues: Dict, send_message_func, stream_tts_func):
        """
        Initialize the enrollment recording manager.
        
        Args:
            db_connection: Database connection
            config: Configuration dict with paths, URLs, etc.
            client_queues: Global client_queues dict
            send_message_func: Function to send messages to clients
            stream_tts_func: Function to stream TTS audio
        """
        self.con = db_connection
        self.config = config
        self.client_queues = client_queues
        self.send_message_to_client = send_message_func
        self.stream_tts_audio = stream_tts_func
        
        # Configuration
        self.max_silence_duration = config.get('enrollment_timeout', 7.0)
        self.reminder_interval = config.get('enrollment_reminder_interval', 2.0)
        self.min_match_threshold = config.get('enrollment_min_match', 0.90)
        self.max_off_topic_utterances = config.get('enrollment_max_off_topic', 3)
    
    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================
    
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
        await self.send_message_to_client(client_id, json.dumps(message))
        
        # Optional TTS
        if not self.config.get('clientSideTTS', False):
            asyncio.create_task(self.stream_tts_audio(client_id, prompt_text))
        
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
            return await self._abort_recording(
                client_id,
                "Enrollment cancelled.",
                use_tts=True
            )
        
        # Check for other speaker
        if speaker_match != "nomatch":
            print(f"[Enrollment] Other speaker detected: {speaker_match}")
            return await self._abort_recording(
                client_id,
                "Another person's voice was detected. Please try again.",
                use_tts=True
            )
        
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
                return await self._abort_recording(
                    client_id,
                    "Too many off-topic responses. Please try again.",
                    use_tts=True
                )
        
        return None  # Still recording
    
    async def check_timeout(self, client_id: str, last_utterance_time: float) -> Optional[str]:
        """
        Check if recording has timed out due to silence.
        
        This should be called from the main loop during silence.
        
        Args:
            client_id: The client's ID
            last_utterance_time: Timestamp of last utterance completion (CLIENT-LEVEL)
            
        Returns:
            'aborted' if timeout occurred, None otherwise
        """
        if client_id not in self.client_queues:
            return None
        
        if "enrollment_state" not in self.client_queues[client_id]:
            return None
        
        enrollment_state = self.client_queues[client_id]["enrollment_state"]
        
        if not enrollment_state["recording_active"]:
            return None
        
        current_time = asyncio.get_event_loop().time()
        silence_duration = current_time - last_utterance_time
        
        if silence_duration >= self.max_silence_duration:
            print(f"[Enrollment] Timeout after {silence_duration:.1f}s of silence")
            return await self._abort_recording(
                client_id,
                "Recording timed out due to inactivity.",
                use_tts=True
            )
        
        return None
    
    async def send_reminder_if_needed(self, client_id: str, last_utterance_time: float) -> None:
        """
        Send reminder prompt if enough silence has passed.
        
        This should be called from the main loop during silence.
        
        Args:
            client_id: The client's ID
            last_utterance_time: Timestamp of last utterance completion (CLIENT-LEVEL)
        """
        if client_id not in self.client_queues:
            return
        
        if "enrollment_state" not in self.client_queues[client_id]:
            return
        
        enrollment_state = self.client_queues[client_id]["enrollment_state"]
        
        if not enrollment_state["recording_active"]:
            return
        
        current_time = asyncio.get_event_loop().time()
        silence_duration = current_time - last_utterance_time
        
        # Only send reminder if enough silence has passed
        if silence_duration < self.reminder_interval:
            return
        
        last_prompt_time = enrollment_state.get("last_prompt_time")
        
        # Don't send if we recently sent a prompt
        if last_prompt_time is not None:
            time_since_prompt = current_time - last_prompt_time
            if time_since_prompt < self.reminder_interval:
                return
        
        # Send reminder
        pangram_text = enrollment_state["pangram_text"]
        reminder_text = f"Still there? Please read: {pangram_text}"
        
        message = {
            "type": "enrollment_reminder",
            "message": reminder_text
        }
        await self.send_message_to_client(client_id, json.dumps(message))
        
        enrollment_state["last_prompt_time"] = current_time
        
        print(f"[Enrollment] Reminder sent after {silence_duration:.1f}s silence")
    
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
            await self.send_message_to_client(client_id, json.dumps(message))
            
            if not self.config.get('clientSideTTS', False):
                asyncio.create_task(self.stream_tts_audio(client_id, success_text))
            
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
        """Abort the enrollment recording."""
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
            await self.send_message_to_client(client_id, json.dumps(abort_message))
            
            if use_tts and not self.config.get('clientSideTTS', False):
                asyncio.create_task(self.stream_tts_audio(client_id, message))
            
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
    
    async def update_enrollment_status(self, request: dict) -> dict:
        """
        Update enrollment status (for Rasa to reset flags).
        
        Args:
            request: Dict with client_id and status
            
        Returns:
            Dict with status
        """
        # This can be used by Rasa to manually update status if needed
        client_id = request.get("client_id")
        status = request.get("status")
        
        if not client_id:
            return {"error": "client_id required"}
        
        # Implementation depends on your needs
        return {"status": "updated"}


# =============================================================================
# INTEGRATION WITH MAIN LOOP
# =============================================================================

"""
INTEGRATION INSTRUCTIONS:

1. Initialize in main():
   
   enrollment_recording_manager = EnrollmentRecordingManager(
       db_connection=con,
       config=CONFIG,
       client_queues=client_queues,
       send_message_func=send_message_to_client,
       stream_tts_func=stream_tts_audio
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
               # NOTE: Do NOT call process_audio_chunk here!
               # We process utterance-by-utterance, not chunk-by-chunk
           
           # === SILENCE SECTION ===
           else:
               silence_counter += 1
               
               # Check enrollment timeout (runs every silence chunk)
               if "enrollment_state" in client_queues[client_id]:
                   if client_queues[client_id]["enrollment_state"]["recording_active"]:
                       result = await enrollment_recording_manager.check_timeout(
                           client_id,
                           last_utterance_time
                       )
                       
                       if result == 'aborted':
                           # Recording was aborted due to timeout
                           pass
                       
                       # Send reminders if needed
                       await enrollment_recording_manager.send_reminder_if_needed(
                           client_id,
                           last_utterance_time
                       )
               
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

4. Add Rasa actions (in actions.py):
   
   class ActionStartEnrollmentRecording(Action):
       def name(self) -> Text:
           return "action_start_enrollment_recording"
       
       async def run(self, dispatcher, tracker, domain):
           client_id = tracker.sender_id
           
           # Get speaker info from slots
           uid = tracker.get_slot("ecapa_uid")
           firstname = tracker.get_slot("ecapa_firstname")
           surname = tracker.get_slot("ecapa_surname")
           
           # Call FastAPI
           response = await call_fastapi_record_pangram(
               client_id=client_id,
               uid=uid,
               firstname=firstname,
               surname=surname
           )
           
           if response.get("status") == "started":
               return [SlotSet("enrollment_active", True)]
           else:
               dispatcher.utter_message(text="Failed to start recording.")
               return []
   
   class ActionResetEnrollmentFlags(Action):
       def name(self) -> Text:
           return "action_reset_enrollment_flags"
       
       def run(self, dispatcher, tracker, domain):
           return [SlotSet("enrollment_active", False)]

5. Add Rasa rules (in rules.yml):
   
   - rule: Reset enrollment flags on completion
     steps:
       - intent: system_enrollment_complete_success
       - action: action_reset_enrollment_flags
   
   - rule: Reset enrollment flags on abort
     steps:
       - intent: system_enrollment_complete_aborted
       - action: action_reset_enrollment_flags

"""


# =============================================================================
# IMPLEMENTATION CHECKLIST
# =============================================================================

"""
COMPLETE IMPLEMENTATION CHECKLIST:

✅ 1. CREATE/UPDATE THREE CLASSES in server01e.py
   [ ] Copy EnrollmentTextUtils class (all static methods)
   [ ] Copy EnrollmentRecordingManager class (core logic)
   [ ] Update EnrollmentAPIHandler class (or create new)

✅ 2. INITIALIZE CLASSES in main()
   [ ] Create enrollment_recording_manager instance
   [ ] Create enrollment_api_handler instance
   [ ] Pass correct dependencies (db, config, client_queues, etc.)

✅ 3. UPDATE FASTAPI ENDPOINT
   [ ] Update /api/record_pangram to use enrollment_api_handler
   [ ] Ensure endpoint returns immediately (non-blocking)

✅ 4. ADD CLIENT-LEVEL last_utterance_time
   [ ] Initialize as local variable in process_audio_from_queue
   [ ] Update when utterance completes (after silence threshold)
   [ ] Pass to check_timeout() and send_reminder_if_needed()

✅ 5. UPDATE MAIN LOOP (process_audio_from_queue)
   [ ] Add timeout check in silence section (calls check_timeout())
   [ ] Add reminder check in silence section (calls send_reminder_if_needed())
   [ ] Call process_utterance() ONLY when utterance completes (during silence)
   [ ] Pass full utterance bytes, transcript, and speaker_match
   [ ] Skip Rasa when enrollment_state["recording_active"] is True

✅ 6. UPDATE RASA ACTIONS (actions.py)
   [ ] Create/update ActionStartEnrollmentRecording
   [ ] Create ActionResetEnrollmentFlags
   [ ] Call FastAPI /api/record_pangram from action

✅ 7. UPDATE RASA CONFIG
   [ ] Add system_enrollment_complete_success intent to nlu.yml
   [ ] Add system_enrollment_complete_aborted intent to nlu.yml
   [ ] Add enrollment_active slot to domain.yml
   [ ] Add rules to reset enrollment flags in rules.yml

✅ 8. ADD CONFIG VALUES
   [ ] enrollment_timeout: 7.0 (seconds)
   [ ] enrollment_reminder_interval: 2.0 (seconds)
   [ ] enrollment_min_match: 0.90 (90% match threshold)
   [ ] enrollment_max_off_topic: 3 (max off-topic utterances)
   [ ] samples_path: (path to save WAV files)
   [ ] rasa_url: (Rasa server URL)

✅ 9. TEST SCENARIOS
   [ ] Recording completes successfully (matches pangram)
   [ ] Timeout abort (7 seconds of silence)
   [ ] Other speaker abort (ECAPA detects different person)
   [ ] Cancel keyword abort ("stop", "cancel", etc.)
   [ ] Reminder prompts (every 2 seconds during silence)
   [ ] Off-topic utterances (max 3 before abort)
   [ ] Multiple clients recording simultaneously

✅ 10. VERIFY
   [ ] WAV files saved with correct filename format
   [ ] Database updated (pangrams marked as recited)
   [ ] Rasa notified on completion/abort
   [ ] enrollment_active flag reset properly
   [ ] No blocking or timeout issues
   [ ] Multi-client support working

NOTES:
- last_utterance_time is CLIENT-LEVEL (not in enrollment_state)
- Process UTTERANCE-BY-UTTERANCE (not chunk-by-chunk)
- Timeout and reminders run in main loop (during silence)
- EnrollmentRecordingManager handles all recording logic
- EnrollmentAPIHandler only handles API interface
- EnrollmentTextUtils provides pure text processing functions
"""
