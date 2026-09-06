"""
ENROLLMENT RECORDING - FINAL CLEAN IMPLEMENTATION
==================================================

ZERO TIMING LOGIC IN ENROLLMENT_STATE OR MANAGER
All timing handled in outer loop by user.

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
    """Text processing utilities - all static methods."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def calculate_fuzzy_match(spoken_text: str, target_text: str) -> float:
        """Calculate fuzzy match score between 0.0 and 1.0."""
        spoken_normalized = EnrollmentTextUtils.normalize_text(spoken_text)
        target_normalized = EnrollmentTextUtils.normalize_text(target_text)
        
        matcher = SequenceMatcher(None, spoken_normalized, target_normalized)
        return matcher.ratio()
    
    @staticmethod
    def is_utterance_on_topic(utterance: str, pangram_text: str, threshold: float = 0.3) -> bool:
        """Check if utterance contains words from pangram."""
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
        """Check if transcript contains cancel keywords."""
        cancel_keywords = [
            "cancel", "stop", "quit", "nevermind", "never mind",
            "abort", "forget it", "end", "exit"
        ]
        
        transcript_lower = transcript.lower()
        return any(keyword in transcript_lower for keyword in cancel_keywords)


# =============================================================================
# CLASS 2: ENROLLMENT RECORDING MANAGER (Core logic, NO TIMING)
# =============================================================================

class EnrollmentRecordingManager:
    """
    Manages enrollment recording sessions.
    
    NO TIMING LOGIC - all timing handled in outer loop.
    """
    
    def __init__(self, db_connection, config: Dict, client_queues: Dict):
        """
        Initialize the enrollment recording manager.
        
        Args:
            db_connection: Database connection
            config: Configuration dict
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
        
        # Initialize enrollment state (NO TIMING FIELDS)
        self.client_queues[client_id]["enrollment_state"] = {
            "recording_active": True,
            "audio_buffer": [],
            "transcript_buffer": [],
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
        data_to_send = {
            "speaker": server_name,
            "speaker_confidence": "certain",
            "final": "True",
            "transcript": pangram_text,
            "asr_confidence": "certain"
        }
        await send_message_to_client(client_id, json.dumps(data_to_send))
        
        return {
            "status": "started",
            "pangram_id": pangram_id,
            "pangram_text": pangram_text
        }
    
    async def process_utterance(
        self,
        client_id: str,
        utterance_audio: bytes,
        utterance_transcript: str
    ) -> Optional[str]:
        """
        Process a completed utterance during enrollment recording.
        
        Args:
            client_id: The client's ID
            utterance_audio: Complete audio for this utterance (bytes)
            utterance_transcript: Transcription of the utterance
            
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
    
    # PRIVATE METHODS =========================================================================
    
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
    
    async def _abort_recording(self, client_id: str, message: str, use_tts: bool = True) -> str:
        """Internal abort method."""
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
            
            # Notify user - send transcript message to client
            data_to_send = {
                "speaker": server_name,
                "speaker_confidence": "certain",
                "final": "True",
                "transcript": message,
                "asr_confidence": "certain"
            }
            json_string = json.dumps(data_to_send)
            await send_message_to_client(client_id, json_string)
            # Handle TTS if not using client-side TTS
            if not clientSideTTS and active_websockets:
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
            wav_file.setsampwidth(2)
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
# CLASS 3: API HANDLER
# =============================================================================

class EnrollmentAPIHandler:
    """Handles FastAPI endpoints for enrollment workflows."""
    
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
        """Query speaker information by UID or name."""
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
        
        Args:
            request: Dict with client_id, uid, firstname, surname
            
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
# SUMMARY
# =============================================================================

"""
ENROLLMENT_STATE STRUCTURE (FINAL - NO TIMING):

{
    "recording_active": bool,
    "audio_buffer": List[numpy.ndarray],
    "transcript_buffer": List[str],
    "pangram_id": int,
    "pangram_text": str,
    "uid": Optional[int],
    "firstname": Optional[str],
    "surname": Optional[str],
    "off_topic_count": int
}

NO TIMING FIELDS:
❌ start_time - removed
❌ last_prompt_time - removed
❌ last_reminder_num - removed

ALL timing logic handled in outer loop using client-level last_utterance_time.

PUBLIC METHODS:
- start_recording(client_id, uid, firstname, surname)
- process_utterance(client_id, utterance_audio, utterance_transcript, speaker_match)
- abort_recording(client_id, reason)

The manager is now focused purely on recording logic with zero timing concerns.
"""
