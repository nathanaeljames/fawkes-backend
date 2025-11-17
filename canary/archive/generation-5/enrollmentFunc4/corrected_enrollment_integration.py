"""
CORRECTED ENROLLMENT INTEGRATION
==================================

Based on actual server01e.py architecture:
- Audio accumulates in current_utterance_buffer during voice active
- Final transcription and ECAPA results available during silence
- Should process UTTERANCE-BY-UTTERANCE, not chunk-by-chunk

Two integration points:
1. During voice active: Check for other speaker (abort condition only)
2. During silence: Process full utterance (audio + transcript + speaker ID)
"""

# =============================================================================
# INTEGRATION POINT 1: During Voice Active (Line ~2619)
# =============================================================================

# In process_audio_from_queue(), within the "if is_voice_active_in_chunk:" block
# Add this AFTER the ECAPA check (around line 2655):

if is_voice_active_in_chunk:
    # ... existing code ...
    
    # Check if we should extract ECAPA embedding
    if ecapa_processor.should_extract_now(len(current_utterance_buffer)):
        ecapa_result = await ecapa_processor.extract_and_match_from_buffer(
            current_utterance_buffer, 
            reason="scheduled"
        )
        if "error" not in ecapa_result:
            print(f"[Speaker ID] {ecapa_result['speaker_result']}")
            SPEAKER = ecapa_result['speaker_result']
            SPEAKER_CONFIDENCE = ecapa_result['speaker_confidence']
            
            # === NEW: CHECK FOR OTHER SPEAKER DURING ENROLLMENT ===
            if "enrollment_state" in client_queues[client_id]:
                if client_queues[client_id]["enrollment_state"]["recording_active"]:
                    # Check if different speaker detected
                    enrollment_state = client_queues[client_id]["enrollment_state"]
                    expected_uid = enrollment_state["uid"]
                    detected_uid = ecapa_result.get('uid_result')
                    
                    # Abort conditions:
                    # 1. New speaker enrollment - ANY positive match is a problem
                    if expected_uid is None and detected_uid is not None and SPEAKER_CONFIDENCE == "certain":
                        print(f"[Enrollment] ABORT: Other speaker detected (UID {detected_uid})")
                        await enrollment_recording_manager.abort_recording(
                            client_id=client_id,
                            reason="other_speaker"
                        )
                    
                    # 2. Existing speaker - DIFFERENT UID detected
                    elif expected_uid is not None and detected_uid != expected_uid and SPEAKER_CONFIDENCE == "certain":
                        print(f"[Enrollment] ABORT: Wrong speaker (expected {expected_uid}, got {detected_uid})")
                        await enrollment_recording_manager.abort_recording(
                            client_id=client_id,
                            reason="other_speaker"
                        )
    
    # ... rest of existing code (send interim results) ...


# =============================================================================
# INTEGRATION POINT 2: During Silence (Line ~2666)
# =============================================================================

# In process_audio_from_queue(), within the "else: # VAD indicates silence" block
# Add this AFTER final ECAPA extraction but BEFORE sending to Rasa (around line 2714):

else: # VAD indicates silence
    silence_counter += 1
    if is_speaking and silence_counter >= SILENCE_CHUNKS_THRESHOLD:
        print("Acoustic finality detected. Processing full utterance...")
        
        # Extract final ECAPA embedding before clearing buffer
        if len(current_utterance_buffer) > 0:
            final_ecapa_result = await ecapa_processor.extract_and_match_from_buffer(
                current_utterance_buffer,
                reason="silence"
            )
            if "error" not in final_ecapa_result:
                print(f"[Final Speaker ID] {final_ecapa_result['speaker_result']}")
                SPEAKER = final_ecapa_result['speaker_result']
                SPEAKER_CONFIDENCE = final_ecapa_result['speaker_confidence']
                nomatch_score = final_ecapa_result['nomatch_score']
                confidence = final_ecapa_result['confidence']
                speaker_uid = final_ecapa_result['uid_result']
        
        # ... Canary-Qwen section (commented out) ...
        
        # === NEW: PROCESS ENROLLMENT UTTERANCE ===
        if "enrollment_state" in client_queues[client_id]:
            if client_queues[client_id]["enrollment_state"]["recording_active"]:
                # Process the FULL utterance for enrollment
                result = await enrollment_recording_manager.process_utterance(
                    client_id=client_id,
                    utterance_audio=current_utterance_buffer,  # Full audio buffer
                    utterance_transcript=final_transcription_text,  # Final transcript
                    speaker_name=SPEAKER,
                    speaker_uid=speaker_uid,
                    speaker_confidence=SPEAKER_CONFIDENCE
                )
                
                if result in ['success', 'aborted']:
                    print(f"[Enrollment] Recording completed: {result}")
                    # Don't send this utterance to Rasa
                    # Reset buffer and continue
                    is_speaking = False
                    silence_counter = 0
                    current_utterance_buffer = b''
                    text = ""
                    SPEAKER = CONFIG["default_speaker"]
                    SPEAKER_CONFIDENCE = CONFIG["default_speaker_confidence"]
                    ecapa_processor.reset_for_new_utterance()
                    continue  # Skip Rasa processing
        
        # === EXISTING: Send to client and Rasa ===
        if final_transcription_text:
            # Only send if NOT in enrollment recording
            if "enrollment_state" not in client_queues[client_id] or \
               not client_queues[client_id]["enrollment_state"].get("recording_active", False):
                
                data_to_send = {
                    "speaker": SPEAKER,
                    "speaker_confidence": SPEAKER_CONFIDENCE,
                    "final": True,
                    "transcript": final_transcription_text,
                    "asr_confidence": ASR_CONFIDENCE
                }
                json_string = json.dumps(data_to_send)
                await send_message_to_client(client_id, json_string)
                
                # Send final utterance to Rasa
                await handle_final_utterance_with_rasa(
                    client_id, 
                    final_transcription_text, 
                    SPEAKER, 
                    speaker_uid, 
                    confidence, 
                    nomatch_score
                )
        
        # ... rest of existing code (suggest_enrollment, reset state) ...


# =============================================================================
# NEW METHOD: process_utterance (replaces process_audio_chunk)
# =============================================================================

class EnrollmentRecordingManager:
    # ... existing methods ...
    
    async def process_utterance(
        self,
        client_id: str,
        utterance_audio: bytes,  # FULL utterance audio buffer
        utterance_transcript: str,  # Final transcript
        speaker_name: str,
        speaker_uid: Optional[int],
        speaker_confidence: str
    ) -> Optional[str]:
        """
        Process a complete utterance during enrollment recording.
        Called from silence section after utterance finality is detected.
        
        Args:
            client_id: Session ID
            utterance_audio: Full utterance audio buffer (bytes, 16-bit PCM)
            utterance_transcript: Final transcription of utterance
            speaker_name: ECAPA identified speaker name
            speaker_uid: ECAPA identified speaker UID
            speaker_confidence: ECAPA confidence level
            
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
        
        # === 1. UPDATE TIMING ===
        enrollment_state["last_speech_time"] = current_time
        
        # === 2. ADD FULL UTTERANCE AUDIO TO BUFFER ===
        # Convert bytes to int16 array for storage
        audio_int16 = np.frombuffer(utterance_audio, dtype=np.int16)
        enrollment_state["audio_buffer"].append(audio_int16)
        print(f"[Enrollment] Added utterance: {len(audio_int16)} samples")
        
        # === 3. CHECK FOR CANCEL COMMAND ===
        if utterance_transcript and self.text_utils.is_cancel_command(utterance_transcript):
            print("[Enrollment] Cancel command detected")
            return await self._abort_recording(
                client_id,
                "Aborting imprint, please try again later.",
                use_tts=True
            )
        
        # === 4. PROCESS TRANSCRIPT ===
        if utterance_transcript:
            pangram_text = enrollment_state["pangram_text"]
            
            # Check if on-topic
            if self.text_utils.is_utterance_on_topic(utterance_transcript, pangram_text):
                enrollment_state["transcript_buffer"].append(utterance_transcript)
                print(f"[Enrollment] On-topic utterance: '{utterance_transcript}'")
                
                # Check completion
                combined = " ".join(enrollment_state["transcript_buffer"])
                match_score = self.text_utils.calculate_fuzzy_match(combined, pangram_text)
                
                print(f"[Enrollment] Match score: {match_score:.2%}")
                
                if match_score >= self.MIN_MATCH_THRESHOLD:
                    print(f"[Enrollment] Pangram complete ({match_score:.2%})")
                    return await self._complete_recording(client_id, ecapa_processor)
            else:
                # Off-topic
                enrollment_state["off_topic_count"] += 1
                print(f"[Enrollment] Off-topic (count: {enrollment_state['off_topic_count']})")
                
                if enrollment_state["off_topic_count"] >= self.MAX_OFF_TOPIC_UTTERANCES:
                    print("[Enrollment] Too many off-topic utterances")
                    return await self._abort_recording(
                        client_id,
                        "Aborting imprint, please try again later.",
                        use_tts=True
                    )
        
        # === 5. CHECK TIMEOUT ===
        silence_duration = current_time - enrollment_state["last_speech_time"]
        
        if silence_duration >= self.SILENCE_TIMEOUT:
            print(f"[Enrollment] Timeout: {silence_duration:.1f}s")
            return await self._abort_recording(
                client_id,
                "Aborting imprint, please try again later.",
                use_tts=True
            )
        
        # === 6. SEND REMINDER IF NEEDED ===
        await self._send_reminder_if_needed(
            client_id, silence_duration, enrollment_state, current_time
        )
        
        return None  # Continue recording
    
    async def abort_recording(
        self,
        client_id: str,
        reason: str = "other_speaker"
    ) -> str:
        """
        Abort recording (can be called from voice active section).
        
        Args:
            client_id: Session ID
            reason: Reason for abort ('other_speaker', etc.)
        """
        messages = {
            "other_speaker": "Aborting imprint, please try again later with no other speakers present.",
            "timeout": "Aborting imprint, please try again later.",
            "off_topic": "Aborting imprint, please try again later.",
            "cancel": "Aborting imprint, please try again later."
        }
        
        message = messages.get(reason, "Aborting imprint, please try again later.")
        
        return await self._abort_recording(client_id, message, use_tts=True)
    
    # ... rest of existing methods remain the same ...


# =============================================================================
# SUMMARY OF CHANGES
# =============================================================================

"""
KEY DIFFERENCES FROM CHUNK-BY-CHUNK APPROACH:

OLD (Wrong):
- Called enrollment handler on every audio chunk
- Added chunks one-by-one
- Processed interim transcripts
- Heavy overhead

NEW (Correct):
- Check for other speaker during voice active (abort only)
- Process full utterance during silence
- Add full utterance audio buffer at once
- Process final transcript
- Matches your actual architecture

BENEFITS:
✅ Processes utterance-by-utterance (not chunk-by-chunk)
✅ Has access to final transcript (not interim)
✅ Has full utterance audio buffer
✅ Has final ECAPA results
✅ Much cleaner integration
✅ Minimal overhead
✅ Matches your VAD-based architecture

TIMING:
- Timestamps still work (updated when utterance processed)
- Timeout checked on each utterance
- Reminders sent appropriately
"""
