# ENROLLMENT RECORDING IMPLEMENTATION CHECKLIST

## Overview
This checklist covers implementing the 3-class enrollment recording system with proper separation of concerns.

---

## Phase 1: Add Classes to server01e.py

### Step 1.1: Add EnrollmentTextUtils class
```python
# In server01e.py, add this class (copy from enrollment_recording_complete.py)

class EnrollmentTextUtils:
    """Static text processing utilities"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        # ... implementation ...
    
    @staticmethod
    def calculate_fuzzy_match(spoken_text: str, target_text: str) -> float:
        # ... implementation ...
    
    @staticmethod
    def is_utterance_on_topic(utterance: str, pangram_text: str, threshold: float = 0.3) -> bool:
        # ... implementation ...
    
    @staticmethod
    def is_cancel_command(transcript: str) -> bool:
        # ... implementation ...
```

**Status**: [ ]

---

### Step 1.2: Add EnrollmentRecordingManager class
```python
# In server01e.py, add this class (copy from enrollment_recording_complete.py)

class EnrollmentRecordingManager:
    """Core recording logic and state management"""
    
    def __init__(self, db_connection, config, client_queues):
        # ... implementation ...
    
    # PUBLIC METHODS
    async def start_recording(self, client_id, uid, firstname, surname):
        # ... implementation ...
    
    async def process_utterance(self, client_id, utterance_audio, utterance_transcript, speaker_match):
        # ... implementation ...
    
    async def check_timeout(self, client_id, last_utterance_time):
        # ... implementation ...
    
    async def send_reminder_if_needed(self, client_id, last_utterance_time):
        # ... implementation ...
    
    # PRIVATE METHODS
    async def _complete_recording(self, client_id):
        # ... implementation ...
    
    async def _abort_recording(self, client_id, message, use_tts):
        # ... implementation ...
    
    async def _save_audio_to_wav(self, client_id, enrollment_state):
        # ... implementation ...
    
    async def _mark_pangram_recited(self, uid, pangram_id):
        # ... implementation ...
    
    async def _notify_rasa(self, client_id, status):
        # ... implementation ...
    
    async def _select_pangram(self, uid):
        # ... implementation ...
```

**Status**: [ ]

---

### Step 1.3: Update EnrollmentAPIHandler class
```python
# In server01e.py, UPDATE your existing EnrollmentAPIHandler
# (or create new if you don't have one)

class EnrollmentAPIHandler:
    """API interface for enrollment endpoints"""
    
    def __init__(self, db_connection, recording_manager):
        self.con = db_connection
        self.recording_manager = recording_manager
    
    async def query_speaker(self, request: dict) -> dict:
        # ... implementation ...
    
    async def record_pangram(self, request: dict) -> dict:
        # Delegates to recording_manager.start_recording()
        # ... implementation ...
    
    async def update_enrollment_status(self, request: dict) -> dict:
        # ... implementation ...
```

**Status**: [ ]

---

## Phase 2: Initialize Classes in main()

### Step 2.1: Create enrollment_recording_manager instance
```python
async def main():
    # ... existing setup code ...
    
    # Initialize enrollment recording manager
    enrollment_recording_manager = EnrollmentRecordingManager(
        db_connection=con,
        config=CONFIG,
        client_queues=client_queues
    )
```

**Status**: [ ]

---

### Step 2.2: Create enrollment_api_handler instance
```python
    # Initialize API handler (uses recording manager)
    enrollment_api_handler = EnrollmentAPIHandler(
        db_connection=con,
        recording_manager=enrollment_recording_manager
    )
```

**Status**: [ ]

---

## Phase 3: Update FastAPI Endpoint

### Step 3.1: Update /api/record_pangram endpoint
```python
@app.post("/api/record_pangram")
async def record_pangram_endpoint(request: dict):
    """Start enrollment recording (returns immediately)"""
    return await enrollment_api_handler.record_pangram(request)
```

**Status**: [ ]

---

## Phase 4: Update Main Audio Processing Loop

### Step 4.1: Add CLIENT-LEVEL last_utterance_time
```python
async def process_audio_from_queue(client_id, nemo_transcriber, nemo_vad, canary_qwen_transcriber):
    """Main audio processing loop"""
    
    # ... existing setup ...
    
    # === NEW: CLIENT-LEVEL timestamp (NOT in enrollment_state) ===
    last_utterance_time = asyncio.get_event_loop().time()
    
    current_utterance_buffer = []
    is_speaking = False
    silence_counter = 0
    
    # ... rest of loop ...
```

**Status**: [ ]

---

### Step 4.2: Update VOICE ACTIVE section
```python
    while True:
        audio_chunk = await client_queues[client_id]["incoming_audio"].get()
        current_time = asyncio.get_event_loop().time()
        
        # ... VAD processing ...
        
        # === VOICE ACTIVE SECTION ===
        if voice_active:
            if not is_speaking:
                is_speaking = True
                silence_counter = 0
                print("Voice activity started")
            
            current_utterance_buffer.append(audio_chunk)
            
            # NOTE: Do NOT process enrollment here!
            # Just accumulate audio during voice activity.
```

**Status**: [ ]

---

### Step 4.3: Update SILENCE section - Add timeout check
```python
        # === SILENCE SECTION ===
        else:
            silence_counter += 1
            
            # === NEW: Check enrollment timeout (runs every silence chunk) ===
            if "enrollment_state" in client_queues[client_id]:
                if client_queues[client_id]["enrollment_state"]["recording_active"]:
                    result = await enrollment_recording_manager.check_timeout(
                        client_id,
                        last_utterance_time
                    )
                    
                    if result == 'aborted':
                        # Timeout occurred, recording aborted
                        pass
```

**Status**: [ ]

---

### Step 4.4: Update SILENCE section - Add reminder check
```python
                    # === NEW: Send reminders if needed ===
                    await enrollment_recording_manager.send_reminder_if_needed(
                        client_id,
                        last_utterance_time
                    )
```

**Status**: [ ]

---

### Step 4.5: Update SILENCE section - Process utterance on finality
```python
            # === Process utterance finality ===
            if is_speaking and silence_counter >= SILENCE_CHUNKS_THRESHOLD:
                print("Acoustic finality detected. Processing full utterance...")
                
                # === NEW: Update CLIENT-LEVEL last_utterance_time ===
                last_utterance_time = current_time
                
                # Get final ECAPA result
                final_speaker_match = "nomatch"
                if len(current_utterance_buffer) > 0:
                    ecapa_result = await ecapa_processor.extract_and_match_from_buffer(
                        current_utterance_buffer,
                        reason="silence"
                    )
                    if "error" not in ecapa_result:
                        final_speaker_match = ecapa_result['speaker_result']
                
                # Get final transcript from Canary-Qwen
                # (Your existing transcription code)
                # Uncomment when ready:
                # final_transcription_text = ...
                final_transcription_text = ""  # TODO: Replace with actual transcript
                
                # === NEW: Process enrollment utterance ===
                if "enrollment_state" in client_queues[client_id]:
                    if client_queues[client_id]["enrollment_state"]["recording_active"]:
                        # Combine utterance buffer
                        utterance_bytes = b''.join(current_utterance_buffer)
                        
                        # Process complete utterance
                        result = await enrollment_recording_manager.process_utterance(
                            client_id=client_id,
                            utterance_audio=utterance_bytes,
                            utterance_transcript=final_transcription_text,
                            speaker_match=final_speaker_match
                        )
                        
                        if result in ['success', 'aborted']:
                            # Recording completed or aborted
                            print(f"[Enrollment] Recording {result}")
```

**Status**: [ ]

---

### Step 4.6: Update SILENCE section - Skip Rasa during enrollment
```python
                # === NEW: Skip Rasa if enrollment active ===
                skip_rasa = False
                if "enrollment_state" in client_queues[client_id]:
                    if client_queues[client_id]["enrollment_state"]["recording_active"]:
                        skip_rasa = True
                        print("[Enrollment] Skipping Rasa (enrollment active)")
                
                # Send to Rasa (if not skipped)
                if not skip_rasa:
                    # Your existing Rasa code
                    # ...
                    pass
                
                # Reset for next utterance
                current_utterance_buffer.clear()
                is_speaking = False
```

**Status**: [ ]

---

## Phase 5: Update Rasa Configuration

### Step 5.1: Add intents to nlu.yml
```yaml
# In nlu.yml, add these intents:

- intent: system_enrollment_complete_success
  examples: |
    - enrollment completed successfully
    - enrollment success

- intent: system_enrollment_complete_aborted
  examples: |
    - enrollment was aborted
    - enrollment failed
```

**Status**: [ ]

---

### Step 5.2: Add slot to domain.yml
```yaml
# In domain.yml, add this slot:

slots:
  enrollment_active:
    type: bool
    initial_value: false
    influence_conversation: false
```

**Status**: [ ]

---

### Step 5.3: Add actions to domain.yml
```yaml
# In domain.yml, add these actions:

actions:
  - action_start_enrollment_recording
  - action_reset_enrollment_flags
```

**Status**: [ ]

---

### Step 5.4: Add rules to rules.yml
```yaml
# In rules.yml, add these rules:

- rule: Reset enrollment flags on success
  steps:
    - intent: system_enrollment_complete_success
    - action: action_reset_enrollment_flags

- rule: Reset enrollment flags on abort
  steps:
    - intent: system_enrollment_complete_aborted
    - action: action_reset_enrollment_flags
```

**Status**: [ ]

---

## Phase 6: Update Rasa Actions

### Step 6.1: Create ActionStartEnrollmentRecording
```python
# In actions.py:

class ActionStartEnrollmentRecording(Action):
    def name(self) -> Text:
        return "action_start_enrollment_recording"
    
    async def run(self, dispatcher, tracker, domain):
        client_id = tracker.sender_id
        
        # Get speaker info from slots
        uid = tracker.get_slot("ecapa_uid")
        firstname = tracker.get_slot("ecapa_firstname")
        surname = tracker.get_slot("ecapa_surname")
        
        # Call FastAPI to start recording
        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(
                    "http://localhost:8000/api/record_pangram",
                    json={
                        "client_id": client_id,
                        "uid": uid,
                        "firstname": firstname,
                        "surname": surname
                    },
                    timeout=5
                )
                
                if response.status == 200:
                    result = await response.json()
                    if result.get("status") == "started":
                        return [SlotSet("enrollment_active", True)]
                    else:
                        dispatcher.utter_message(text="Failed to start recording.")
                        return []
                else:
                    dispatcher.utter_message(text="Error contacting server.")
                    return []
                    
        except Exception as e:
            print(f"Error starting enrollment: {e}")
            dispatcher.utter_message(text="An error occurred.")
            return []
```

**Status**: [ ]

---

### Step 6.2: Create ActionResetEnrollmentFlags
```python
# In actions.py:

class ActionResetEnrollmentFlags(Action):
    def name(self) -> Text:
        return "action_reset_enrollment_flags"
    
    def run(self, dispatcher, tracker, domain):
        """Reset enrollment_active flag"""
        return [SlotSet("enrollment_active", False)]
```

**Status**: [ ]

---

## Phase 7: Add Configuration Values

### Step 7.1: Add enrollment config to CONFIG dict
```python
# In server01e.py, add to CONFIG:

CONFIG = {
    # ... existing config ...
    
    # Enrollment recording settings
    'enrollment_timeout': 7.0,              # Seconds of silence before timeout
    'enrollment_reminder_interval': 2.0,    # Seconds between reminders
    'enrollment_min_match': 0.90,           # 90% match threshold for completion
    'enrollment_max_off_topic': 3,          # Max off-topic utterances before abort
    'samples_path': '/path/to/samples',     # Where to save WAV files
    'rasa_url': 'http://localhost:5005',    # Rasa server URL
    'clientSideTTS': False,                 # Whether client handles TTS
}
```

**Status**: [ ]

---

## Phase 8: Test Scenarios

### Test 1: Successful completion
- [ ] Start recording
- [ ] Speak pangram correctly
- [ ] Verify WAV file saved with correct filename
- [ ] Verify database updated (pangram marked as recited)
- [ ] Verify Rasa notified (system_enrollment_complete_success)
- [ ] Verify enrollment_active flag reset

---

### Test 2: Timeout abort
- [ ] Start recording
- [ ] Stay silent for >7 seconds
- [ ] Verify abort message sent
- [ ] Verify Rasa notified (system_enrollment_complete_aborted)
- [ ] Verify state cleaned up

---

### Test 3: Cancel command
- [ ] Start recording
- [ ] Say "cancel" or "stop"
- [ ] Verify recording aborted
- [ ] Verify state cleaned up

---

### Test 4: Other speaker detected
- [ ] Start recording
- [ ] Have different person speak
- [ ] Verify abort due to speaker mismatch
- [ ] Verify state cleaned up

---

### Test 5: Off-topic utterances
- [ ] Start recording
- [ ] Say 3+ off-topic things
- [ ] Verify abort after max off-topic reached
- [ ] Verify state cleaned up

---

### Test 6: Reminders
- [ ] Start recording
- [ ] Stay silent for 2 seconds
- [ ] Verify reminder sent
- [ ] Wait another 2 seconds
- [ ] Verify another reminder sent

---

### Test 7: Multi-client
- [ ] Start recording for client A
- [ ] Start recording for client B
- [ ] Verify both recordings independent
- [ ] Verify no interference between clients

---

## Phase 9: Verify Architecture

### Verify 1: Separation of concerns
- [ ] EnrollmentTextUtils has only static text processing
- [ ] EnrollmentRecordingManager has only recording logic
- [ ] EnrollmentAPIHandler has only API interface

---

### Verify 2: State management
- [ ] last_utterance_time is CLIENT-LEVEL (not in enrollment_state)
- [ ] enrollment_state replaces old enrollment_active flag
- [ ] enrollment_state properly cleaned up on completion/abort

---

### Verify 3: Non-blocking
- [ ] /api/record_pangram returns immediately
- [ ] No blocking waits in any enrollment functions
- [ ] Recording happens in main audio loop

---

### Verify 4: Multi-client safety
- [ ] Each client has own enrollment_state
- [ ] Each client has own last_utterance_time
- [ ] No shared state between clients

---

## Completion Checklist Summary

### Classes Added
- [ ] EnrollmentTextUtils
- [ ] EnrollmentRecordingManager
- [ ] EnrollmentAPIHandler (updated)

### Main Loop Updated
- [ ] Added last_utterance_time (client-level)
- [ ] Added timeout check in silence section
- [ ] Added reminder check in silence section
- [ ] Added process_utterance call on finality
- [ ] Added skip Rasa check during enrollment

### FastAPI Updated
- [ ] Updated /api/record_pangram endpoint

### Rasa Updated
- [ ] Added intents to nlu.yml
- [ ] Added slot to domain.yml
- [ ] Added actions to domain.yml
- [ ] Added rules to rules.yml
- [ ] Created ActionStartEnrollmentRecording
- [ ] Created ActionResetEnrollmentFlags

### Configuration Updated
- [ ] Added enrollment config values

### Testing Completed
- [ ] Successful completion test
- [ ] Timeout abort test
- [ ] Cancel command test
- [ ] Other speaker test
- [ ] Off-topic test
- [ ] Reminders test
- [ ] Multi-client test

### Architecture Verified
- [ ] Separation of concerns
- [ ] State management
- [ ] Non-blocking operation
- [ ] Multi-client safety

---

## Quick Reference

**Key Files**:
- `enrollment_recording_complete.py` - Complete implementation with all 3 classes
- `ANSWERS_TO_QUESTIONS.md` - Answers to your specific questions
- This file - Step-by-step checklist

**Key Concepts**:
- `last_utterance_time` is CLIENT-LEVEL
- Process UTTERANCE-BY-UTTERANCE (not chunk-by-chunk)
- Timeout/reminders run in main loop during silence
- `enrollment_state` replaces `enrollment_active` flag

**Integration Points**:
1. Initialize classes in `main()`
2. Update FastAPI endpoint
3. Update main audio loop
4. Update Rasa config
5. Create Rasa actions

---

**Ready to implement? Start with Phase 1!**
