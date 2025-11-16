"""
ENROLLMENT RECORDING - CLASS ORGANIZATION
==========================================

Proper separation of concerns following SOLID principles:

1. EnrollmentAPIHandler - API endpoints & request/response handling
2. EnrollmentRecordingManager - Recording orchestration & state management  
3. EnrollmentTextUtils - Text processing utilities (fuzzy matching, etc.)

This keeps each class focused on a single responsibility.
"""

import random
import asyncio
import wave
import json
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import aiohttp
import re


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
# CLASS 2: RECORDING MANAGER (Core enrollment logic)
# =============================================================================

class EnrollmentRecordingManager:
    """
    Manages the enrollment recording process.
    
    Responsibilities:
    - Recording orchestration (start, monitor, complete/abort)
    - State management (enrollment_state in client_queues)
    - Audio buffer handling
    - Pangram selection and tracking
    - Integration with ECAPA processor
    - Rasa notification
    
    """
    
    # Recording constants
    SILENCE_TIMEOUT = 7.0  # seconds
    PROMPT_REMINDER_INTERVAL = 2.0  # seconds
    MAX_OFF_TOPIC_UTTERANCES = 3
    MIN_MATCH_THRESHOLD = 0.90  # 90% completion threshold
    
    def __init__(self, db_connection, config: Dict, client_queues: Dict):
        """
        Initialize the recording manager.
        
        Args:
            db_connection: DuckDB connection
            config: Server configuration dict
            client_queues: Reference to client_queues dict
        """
        self.con = db_connection
        self.config = config
        self.client_queues = client_queues
        self.text_utils = EnrollmentTextUtils()
    
    async def start_recording(
        self,
        client_id: str,
        uid: Optional[int] = None,
        firstname: str = None,
        surname: str = None
    ) -> Dict[str, Any]:
        """
        Start enrollment recording for a client.
        Returns immediately after setting up state.
        
        Args:
            client_id: Unique session identifier
            uid: Speaker UID if updating existing speaker, None for new
            firstname: First name of speaker
            surname: Surname of speaker
            
        Returns:
            Dict with status 'started' or 'error'
        """
        try:
            # 1. Select pangram
            pangram_id, pangram_text = await self._select_pangram(uid)
            
            if pangram_id is None:
                return {
                    'status': 'error',
                    'message': 'Failed to select pangram'
                }
            
            # 2. Validate client exists
            if client_id not in self.client_queues:
                return {
                    'status': 'error',
                    'message': 'Client not found'
                }
            
            # 3. Initialize enrollment state
            self.client_queues[client_id]["enrollment_state"] = {
                "recording_active": True,
                "pangram_id": pangram_id,
                "pangram_text": pangram_text,
                "uid": uid,
                "firstname": firstname,
                "surname": surname,
                "audio_buffer": [],
                "transcript_buffer": [],
                "off_topic_count": 0,
                "last_speech_time": asyncio.get_event_loop().time(),
                "start_time": asyncio.get_event_loop().time(),
                "last_prompt_time": None
            }
            
            # 4. Send pangram to client
            await self._send_pangram_to_client(client_id, pangram_text)
            
            print(f"[Enrollment] Started recording for {client_id}, pangram {pangram_id}")
            
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
    
    async def process_audio_chunk(
        self,
        client_id: str,
        audio_chunk: np.ndarray,
        is_voice_active: bool,
        current_transcript: str,
        speaker_name: str,
        speaker_uid: Optional[int],
        speaker_confidence: str,
        ecapa_processor: 'ECAPASpeakerProcessor'
    ) -> Optional[str]:
        """
        Process audio during enrollment recording.
        Called from main audio processing loop.
        
        Args:
            client_id: Session ID
            audio_chunk: Current audio chunk (int16 array)
            is_voice_active: Whether VAD detected voice
            current_transcript: Final transcript (or empty string)
            speaker_name: ECAPA identified speaker name
            speaker_uid: ECAPA identified speaker UID
            speaker_confidence: ECAPA confidence level
            ecapa_processor: ECAPA processor instance
            
        Returns:
            'success', 'aborted', or None (continue recording)
        """
        if client_id not in self.client_queues:
            return 'aborted'
        
        if "enrollment_state" not in self.client_queues[client_id]:
            return None
        
        enrollment_state = self.client_queues[client_id]["enrollment_state"]
        
        if not enrollment_state.get("recording_active", False):
            return None
        
        current_time = asyncio.get_event_loop().time()
        
        # Check for other speaker
        if await self._check_other_speaker(
            client_id, is_voice_active, speaker_uid, 
            speaker_confidence, enrollment_state
        ):
            return await self._abort_recording(
                client_id,
                "Aborting imprint, please try again later with no other speakers present.",
                use_tts=True
            )
        
        # Process audio
        if is_voice_active:
            enrollment_state["audio_buffer"].append(audio_chunk)
            enrollment_state["last_speech_time"] = current_time
        
        # Process transcript
        if current_transcript:
            result = await self._process_transcript(
                client_id, current_transcript, enrollment_state
            )
            if result == 'success':
                return await self._complete_recording(client_id, ecapa_processor)
            elif result == 'aborted':
                return await self._abort_recording(
                    client_id,
                    "Aborting imprint, please try again later.",
                    use_tts=True
                )
        
        # Check timeout
        silence_duration = current_time - enrollment_state["last_speech_time"]
        
        if silence_duration >= self.SILENCE_TIMEOUT:
            print(f"[Enrollment] Timeout: {silence_duration:.1f}s")
            return await self._abort_recording(
                client_id,
                "Aborting imprint, please try again later.",
                use_tts=True
            )
        
        # Send reminder
        await self._send_reminder_if_needed(
            client_id, silence_duration, enrollment_state, current_time
        )
        
        return None
    
    # -------------------------------------------------------------------------
    # PRIVATE METHODS - Internal implementation
    # -------------------------------------------------------------------------
    
    async def _select_pangram(self, uid: Optional[int]) -> Tuple[Optional[int], Optional[str]]:
        """Select an unrecited pangram for the speaker"""
        try:
            all_pangrams = self.con.execute("""
                SELECT id, text FROM pangrams ORDER BY id
            """).fetchall()
            
            if not all_pangrams:
                print("[Enrollment] Error: No pangrams in database")
                return None, None
            
            if uid is None:
                selected = random.choice(all_pangrams)
                return selected[0], selected[1]
            
            recited = self.con.execute("""
                SELECT pangrams FROM speakers WHERE uid = ?
            """, [uid]).fetchone()
            
            if recited is None:
                selected = random.choice(all_pangrams)
                return selected[0], selected[1]
            
            recited_ids = recited[0] if recited[0] else []
            
            available = [p for p in all_pangrams if p[0] not in recited_ids]
            
            if available:
                selected = random.choice(available)
            else:
                selected = random.choice(all_pangrams)
            
            print(f"[Enrollment] Selected pangram {selected[0]} for UID {uid}")
            return selected[0], selected[1]
            
        except Exception as e:
            print(f"[Enrollment] Error selecting pangram: {e}")
            return None, None
    
    async def _send_pangram_to_client(self, client_id: str, pangram_text: str):
        """Send pangram text to client"""
        from server_globals import send_message_to_client  # Import to avoid circular dependency
        
        message = {
            "type": "enrollment_prompt",
            "text": pangram_text
        }
        await send_message_to_client(client_id, json.dumps(message))
    
    async def _check_other_speaker(
        self,
        client_id: str,
        is_voice_active: bool,
        speaker_uid: Optional[int],
        speaker_confidence: str,
        enrollment_state: Dict
    ) -> bool:
        """Check if a different speaker was detected"""
        if not is_voice_active or speaker_confidence != "certain":
            return False
        
        expected_uid = enrollment_state["uid"]
        
        # New speaker enrollment - any match is a problem
        if expected_uid is None and speaker_uid is not None:
            print(f"[Enrollment] Other speaker detected (UID: {speaker_uid})")
            return True
        
        # Existing speaker - different UID detected
        if expected_uid is not None and speaker_uid != expected_uid:
            print(f"[Enrollment] Wrong speaker (expected {expected_uid}, got {speaker_uid})")
            return True
        
        return False
    
    async def _process_transcript(
        self,
        client_id: str,
        transcript: str,
        enrollment_state: Dict
    ) -> Optional[str]:
        """
        Process a new transcript.
        Returns 'success', 'aborted', or None
        """
        print(f"[Enrollment] Transcript: '{transcript}'")
        
        # Check for cancel
        if self.text_utils.is_cancel_command(transcript):
            print("[Enrollment] Cancel command detected")
            return 'aborted'
        
        pangram_text = enrollment_state["pangram_text"]
        
        # Check if on-topic
        if self.text_utils.is_utterance_on_topic(transcript, pangram_text):
            enrollment_state["transcript_buffer"].append(transcript)
            
            # Check completion
            combined = " ".join(enrollment_state["transcript_buffer"])
            match_score = self.text_utils.calculate_fuzzy_match(combined, pangram_text)
            
            print(f"[Enrollment] Match score: {match_score:.2%}")
            
            if match_score >= self.MIN_MATCH_THRESHOLD:
                print(f"[Enrollment] Pangram complete ({match_score:.2%})")
                return 'success'
        else:
            # Off-topic
            enrollment_state["off_topic_count"] += 1
            print(f"[Enrollment] Off-topic (count: {enrollment_state['off_topic_count']})")
            
            if enrollment_state["off_topic_count"] >= self.MAX_OFF_TOPIC_UTTERANCES:
                print("[Enrollment] Too many off-topic utterances")
                return 'aborted'
        
        return None
    
    async def _send_reminder_if_needed(
        self,
        client_id: str,
        silence_duration: float,
        enrollment_state: Dict,
        current_time: float
    ):
        """Send reminder to continue reciting if paused"""
        if silence_duration < self.PROMPT_REMINDER_INTERVAL:
            return
        
        last_prompt = enrollment_state["last_prompt_time"]
        
        if last_prompt is not None:
            if (current_time - last_prompt) < self.PROMPT_REMINDER_INTERVAL:
                return
        
        # Only remind if user has started but not finished
        if len(enrollment_state["transcript_buffer"]) == 0:
            return
        
        from server_globals import send_message_to_client
        
        message = {
            "type": "enrollment_reminder",
            "text": "Please finish reciting the prompt!"
        }
        await send_message_to_client(client_id, json.dumps(message))
        enrollment_state["last_prompt_time"] = current_time
    
    async def _complete_recording(
        self,
        client_id: str,
        ecapa_processor: 'ECAPASpeakerProcessor'
    ) -> str:
        """Complete enrollment successfully"""
        enrollment_state = self.client_queues[client_id]["enrollment_state"]
        
        try:
            # Save WAV
            wav_path = await self._save_audio_to_wav(client_id, enrollment_state)
            print(f"[Enrollment] Saved: {wav_path}")
            
            # Update imprint
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
                    new_speaker = self.con.execute("""
                        SELECT uid FROM speakers 
                        WHERE firstname = ? AND surname = ?
                        ORDER BY uid DESC LIMIT 1
                    """, [firstname, surname]).fetchone()
                    
                    if new_speaker:
                        new_uid = new_speaker[0]
                        await self._mark_pangram_recited(new_uid, enrollment_state["pangram_id"])
            else:
                # Update existing
                success = await ecapa_processor.update_speaker_imprint_from_file(
                    wav_path=str(wav_path),
                    uid=uid
                )
                
                if success:
                    await self._mark_pangram_recited(uid, enrollment_state["pangram_id"])
            
            # Notify user and Rasa
            await self._send_completion_message(client_id, "Enrollment completed successfully!")
            await self._notify_rasa(client_id, 'success')
            
            # Cleanup
            del self.client_queues[client_id]["enrollment_state"]
            
            return 'success'
            
        except Exception as e:
            print(f"[Enrollment] Error completing: {e}")
            return await self._abort_recording(client_id, "Enrollment failed.", use_tts=True)
    
    async def _abort_recording(
        self,
        client_id: str,
        message: str,
        use_tts: bool = True
    ) -> str:
        """Abort enrollment"""
        if client_id not in self.client_queues:
            return 'aborted'
        
        if "enrollment_state" not in self.client_queues[client_id]:
            return 'aborted'
        
        enrollment_state = self.client_queues[client_id]["enrollment_state"]
        
        try:
            # Save WAV for debugging
            wav_path = await self._save_audio_to_wav(client_id, enrollment_state)
            print(f"[Enrollment] Aborted, saved: {wav_path}")
            
            # Notify user and Rasa
            await self._send_abort_message(client_id, message, use_tts)
            await self._notify_rasa(client_id, 'aborted')
            
            # Cleanup
            del self.client_queues[client_id]["enrollment_state"]
            
            return 'aborted'
            
        except Exception as e:
            print(f"[Enrollment] Error during abort: {e}")
            if "enrollment_state" in self.client_queues[client_id]:
                del self.client_queues[client_id]["enrollment_state"]
            return 'aborted'
    
    async def _save_audio_to_wav(
        self,
        client_id: str,
        enrollment_state: Dict
    ) -> Path:
        """Save audio buffer to WAV file"""
        session_id = client_id.replace('-', '')[:8]
        pangram_id = enrollment_state["pangram_id"]
        uid = enrollment_state["uid"]
        firstname = enrollment_state.get("firstname") or "unknown"
        surname = enrollment_state.get("surname") or "unknown"
        
        if uid is not None:
            filename = f"pangram{pangram_id}_{session_id}_{surname}_{firstname}_uid{uid}.wav"
        else:
            filename = f"pangram{pangram_id}_{session_id}_{surname}_{firstname}.wav"
        
        wav_path = Path(self.config['samples_path']) / filename
        
        audio_buffer = enrollment_state["audio_buffer"]
        
        if not audio_buffer:
            concatenated = np.array([], dtype=np.int16)
        else:
            concatenated = np.concatenate(audio_buffer)
        
        # Save using wave module
        with wave.open(str(wav_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(concatenated.tobytes())
        
        return wav_path
    
    async def _mark_pangram_recited(self, uid: int, pangram_id: int):
        """Mark pangram as recited in database"""
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
    
    async def _send_completion_message(self, client_id: str, text: str):
        """Send completion message to client"""
        from server_globals import send_message_to_client, stream_tts_audio, clientSideTTS
        
        message = {"type": "enrollment_complete", "text": text}
        await send_message_to_client(client_id, json.dumps(message))
        
        if not clientSideTTS:
            asyncio.create_task(stream_tts_audio(client_id, text))
    
    async def _send_abort_message(self, client_id: str, text: str, use_tts: bool):
        """Send abort message to client"""
        from server_globals import send_message_to_client, stream_tts_audio, clientSideTTS
        
        message = {"type": "enrollment_aborted", "text": text}
        await send_message_to_client(client_id, json.dumps(message))
        
        if use_tts and not clientSideTTS:
            asyncio.create_task(stream_tts_audio(client_id, text))
    
    async def _notify_rasa(self, client_id: str, status: str):
        """Notify Rasa of completion via trigger_intent"""
        try:
            intent_name = f"system_enrollment_complete_{status}"
            
            async with aiohttp.ClientSession() as session:
                response = await session.post(
                    f"{self.config['rasa_url']}/conversations/{client_id}/trigger_intent",
                    json={"name": intent_name, "entities": []},
                    params={"output_channel": "latest"},
                    timeout=5
                )
                
                if response.status == 200:
                    print(f"[Enrollment] Triggered intent: {intent_name}")
                else:
                    print(f"[Enrollment] Failed to trigger intent: {response.status}")
                    
        except Exception as e:
            print(f"[Enrollment] Error notifying Rasa: {e}")


# =============================================================================
# CLASS 3: API HANDLER (Updated to use EnrollmentRecordingManager)
# =============================================================================

class EnrollmentAPIHandler:
    """
    Handles FastAPI endpoints for speaker enrollment workflows.
    
    Responsibilities:
    - API request/response handling
    - Input validation
    - Database queries
    - Delegation to EnrollmentRecordingManager
    
    Does NOT handle:
    - Recording logic (delegated to EnrollmentRecordingManager)
    - Text processing (delegated to EnrollmentTextUtils)
    """
    
    def __init__(self, db_connection, recording_manager: EnrollmentRecordingManager):
        """
        Initialize the API handler.
        
        Args:
            db_connection: DuckDB connection
            recording_manager: EnrollmentRecordingManager instance
        """
        self.con = db_connection
        self.recording_manager = recording_manager
    
    async def query_speaker(self, request) -> Dict[str, Any]:
        """
        Query speaker information from database.
        Used by Rasa to check if a speaker exists before enrollment.
        """
        try:
            if request.table != "speakers":
                raise HTTPException(status_code=400, detail=f"Unsupported table: {request.table}")
            
            # Case-insensitive comparison
            if request.surname:
                result = self.con.execute("""
                    SELECT uid FROM speakers 
                    WHERE LOWER(firstname) = LOWER(?) AND LOWER(surname) = LOWER(?)
                """, (request.firstname, request.surname)).fetchone()
            else:
                result = self.con.execute("""
                    SELECT uid FROM speakers 
                    WHERE LOWER(firstname) = LOWER(?) AND surname IS NULL
                """, (request.firstname,)).fetchone()
            
            uid = result[0] if result else None
            print(f"[Enrollment API] Query: {request.firstname} {request.surname} → UID={uid}")
            
            return {"uid": uid, "success": True}
        
        except Exception as e:
            print(f"[Enrollment API] Error querying speaker: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def record_pangram(self, request) -> Dict[str, Any]:
        """
        Initiate pangram recording.
        Delegates to EnrollmentRecordingManager.
        """
        try:
            uid = int(request.uid) if request.uid else None
            firstname = request.get('firstname')
            surname = request.get('surname')
            client_id = request.get('client_id')
            
            if not client_id:
                return {"status": "error", "message": "client_id required"}
            
            # Delegate to recording manager
            result = await self.recording_manager.start_recording(
                client_id=client_id,
                uid=uid,
                firstname=firstname,
                surname=surname
            )
            
            return result
        
        except Exception as e:
            print(f"[Enrollment API] Error in record_pangram: {e}")
            return {"status": "error", "message": str(e)}
    
    async def update_enrollment_status(self, request) -> Dict[str, Any]:
        """
        Update enrollment status (legacy endpoint).
        May not be needed with new architecture.
        """
        try:
            client_id = request.client_id
            status = request.status
            
            # Strip "client_" prefix if present
            if client_id.startswith("client_"):
                client_id = client_id[7:]
            
            print(f"[Enrollment API] Status update: {client_id} → {status}")
            
            return {"success": True, "message": "Status updated"}
        
        except Exception as e:
            print(f"[Enrollment API] Error updating status: {e}")
            return {"success": False, "message": str(e)}


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

"""
# In server01e.py:

# Initialize classes
enrollment_recording_manager = EnrollmentRecordingManager(
    db_connection=con,
    config=CONFIG,
    client_queues=client_queues
)

enrollment_api_handler = EnrollmentAPIHandler(
    db_connection=con,
    recording_manager=enrollment_recording_manager
)

# FastAPI endpoint
@app.post("/api/record_pangram")
async def record_pangram_endpoint(request: dict):
    return await enrollment_api_handler.record_pangram(request)

# In process_audio_from_queue():
if "enrollment_state" in client_queues[client_id]:
    if client_queues[client_id]["enrollment_state"]["recording_active"]:
        result = await enrollment_recording_manager.process_audio_chunk(
            client_id=client_id,
            audio_chunk=audio_int16,
            is_voice_active=True,
            current_transcript=FINAL_TRANSCRIPT,
            speaker_name=SPEAKER_NAME,
            speaker_uid=SPEAKER_UID,
            speaker_confidence=SPEAKER_CONFIDENCE,
            ecapa_processor=ecapa_processor
        )
"""
