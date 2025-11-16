# SIMPLIFIED ENROLLMENT ARCHITECTURE - Integration Guide

## Overview

Your simplified architecture is **much better** because:

1. ✅ **Uses existing infrastructure** - no reinventing the wheel
2. ✅ **No blocking/timeout issues** - action returns immediately
3. ✅ **In-memory buffering** - write only on completion
4. ✅ **Natural integration** - leverages existing audio loop
5. ✅ **Simpler state management** - everything in client_queues

---

## Architecture Flow

```
[Rasa Action] 
    ↓ (kicks off)
[FastAPI /api/record_pangram] 
    ↓ (sets up state)
[record_pangram() function]
    ↓ (returns immediately with 'started')
[Rasa Action returns]
    ↓ (user is free to interact)
[process_audio_from_queue loop]
    ↓ (processes audio, detects completion)
[complete_enrollment() or abort_enrollment()]
    ↓ (notifies Rasa)
[Rasa receives system message]
    ↓ (resets enrollment_active)
[Done]
```

---

## 1. FastAPI Endpoint (server01e.py)

Update or create the `/api/record_pangram` endpoint:

```python
@app.post("/api/record_pangram")
async def record_pangram_endpoint(request: dict):
    """
    Endpoint for Rasa to trigger pangram recording.
    Returns immediately after starting recording.
    """
    client_id = request.get("client_id")
    uid = request.get("uid")
    firstname = request.get("firstname")
    surname = request.get("surname")
    
    if not client_id:
        raise HTTPException(status_code=400, detail="client_id required")
    
    # Kick off recording (returns immediately)
    result = await record_pangram(
        client_id=client_id,
        uid=uid,
        firstname=firstname,
        surname=surname
    )
    
    # Returns 'started' status immediately
    return result
```

**Key change**: Endpoint returns immediately with `{'status': 'started'}`, doesn't wait for completion.

---

## 2. Rasa Action (actions.py)

Update `ActionStartEnrollmentRecording`:

```python
class ActionStartEnrollmentRecording(Action):
    def name(self) -> Text:
        return "action_start_enrollment_recording"
    
    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        sender_id = tracker.sender_id
        imprint_uid = tracker.get_slot("imprint_uid")
        imprint_firstname = tracker.get_slot("imprint_firstname")
        imprint_surname = tracker.get_slot("imprint_surname")
        
        # Kick off recording on server
        fastapi_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/record_pangram"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    fastapi_url,
                    json={
                        "action": "record_pangram",
                        "client_id": sender_id,
                        "uid": imprint_uid,
                        "firstname": imprint_firstname,
                        "surname": imprint_surname
                    },
                    timeout=5  # Short timeout, just to start recording
                ) as response:
                    
                    if response.status == HTTPStatus.OK:
                        result = await response.json()
                        logger.info(f"Recording started for {sender_id}: {result}")
                        
                        # Keep enrollment_active True
                        # It will be reset when we receive system message
                        return [SlotSet("enrollment_active", True)]
                    else:
                        logger.error(f"Failed to start recording: {response.status}")
                        return [SlotSet("enrollment_active", False)]
                        
        except Exception as e:
            logger.error(f"Error starting enrollment recording: {e}")
            return [SlotSet("enrollment_active", False)]
```

**Key change**: Action returns immediately after starting recording. Rasa continues with `enrollment_active=True`.

---

## 3. Add System Message Handling (Rasa)

### 3.1 Add new intents to nlu.yml

```yaml
- intent: system_enrollment_complete_success
  examples: |
    - SYSTEM_ENROLLMENT_COMPLETE_SUCCESS

- intent: system_enrollment_complete_aborted
  examples: |
    - SYSTEM_ENROLLMENT_COMPLETE_ABORTED
```

### 3.2 Add intents to domain.yml

```yaml
intents:
  # ... existing intents ...
  - system_enrollment_complete_success
  - system_enrollment_complete_aborted
```

### 3.3 Add responses to domain.yml

```yaml
responses:
  utter_enrollment_success:
    - text: "Your voice imprint has been successfully recorded!"
  
  utter_enrollment_aborted:
    - text: "Enrollment was not completed. Please try again when ready."
```

### 3.4 Add rules to rules.yml

```yaml
- rule: Handle enrollment completion - success
  condition:
  - slot_was_set:
    - enrollment_active: true
  steps:
  - intent: system_enrollment_complete_success
  - action: action_reset_enrollment_flags
  - action: utter_enrollment_success

- rule: Handle enrollment completion - aborted
  condition:
  - slot_was_set:
    - enrollment_active: true
  steps:
  - intent: system_enrollment_complete_aborted
  - action: action_reset_enrollment_flags
  - action: utter_enrollment_aborted
```

### 3.5 Add action to actions.py

```python
class ActionResetEnrollmentFlags(Action):
    """Reset enrollment flags when recording completes"""
    
    def name(self) -> Text:
        return "action_reset_enrollment_flags"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info("Resetting enrollment flags after completion")
        
        return [
            SlotSet("enrollment_active", False),
            SlotSet("enrollment_muted", False)
        ]
```

---

## 4. Integration into process_audio_from_queue

Add this to your main audio processing loop in `server01e.py`:

```python
async def process_audio_from_queue(client_id, nemo_transcriber, nemo_vad, canary_qwen_transcriber):
    # ... existing initialization code ...
    
    SPEAKER_NAME = "unknown speaker"
    SPEAKER_UID = None
    SPEAKER_CONFIDENCE = "uncertain"
    FINAL_TRANSCRIPT = ""  # Store final transcript
    
    try:
        while client_id in client_queues:
            try:
                audio_data = await asyncio.wait_for(
                    client_queues[client_id]["incoming_audio"].get(), 
                    timeout=1.0
                )
                
                # ... existing audio processing (convert to int16, etc.) ...
                
                # Process with VAD
                vad_outputs = nemo_vad.process_frame(audio_int16)
                is_voice_active_in_chunk = nemo_vad.is_voice_active(vad_outputs)
                
                if is_voice_active_in_chunk:
                    # ... existing voice processing ...
                    # ... NeMo streaming ASR for interim results ...
                    # ... ECAPA speaker identification ...
                    # Store SPEAKER_NAME, SPEAKER_UID, SPEAKER_CONFIDENCE
                    
                    # When you get a final Canary-Qwen transcript:
                    # FINAL_TRANSCRIPT = canary_result  # or however you get it
                    
                    # =============================================================
                    # NEW: CHECK FOR ACTIVE ENROLLMENT RECORDING
                    # =============================================================
                    if "enrollment_state" in client_queues[client_id]:
                        if client_queues[client_id]["enrollment_state"]["recording_active"]:
                            enrollment_result = await handle_enrollment_recording(
                                client_id=client_id,
                                audio_chunk=audio_int16,
                                is_voice_active=True,
                                current_transcript=FINAL_TRANSCRIPT,  # Empty string if no final yet
                                speaker_name=SPEAKER_NAME,
                                speaker_uid=SPEAKER_UID,
                                speaker_confidence=SPEAKER_CONFIDENCE,
                                ecapa_processor=ecapa_processor
                            )
                            
                            if enrollment_result in ['success', 'aborted']:
                                print(f"[Enrollment] Recording completed: {enrollment_result}")
                                # Clear the final transcript so it's not re-used
                                FINAL_TRANSCRIPT = ""
                                continue
                    
                    # Clear final transcript after processing
                    FINAL_TRANSCRIPT = ""
                    
                else:  # Silence detected by VAD
                    # ... existing silence handling ...
                    
                    # =============================================================
                    # NEW: CHECK ENROLLMENT DURING SILENCE (for timeouts)
                    # =============================================================
                    if "enrollment_state" in client_queues[client_id]:
                        if client_queues[client_id]["enrollment_state"]["recording_active"]:
                            enrollment_result = await handle_enrollment_recording(
                                client_id=client_id,
                                audio_chunk=np.array([], dtype=np.int16),
                                is_voice_active=False,
                                current_transcript="",
                                speaker_name=SPEAKER_NAME,
                                speaker_uid=SPEAKER_UID,
                                speaker_confidence=SPEAKER_CONFIDENCE,
                                ecapa_processor=ecapa_processor
                            )
                            
                            if enrollment_result in ['success', 'aborted']:
                                print(f"[Enrollment] Recording completed: {enrollment_result}")
                                continue
                
                # ... rest of existing processing ...
                
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error processing audio for {client_id}: {e}")
                break
    
    finally:
        print("Async Audio processing stopped")
```

**Key integration points**:
1. Check for `enrollment_state` in client_queues
2. Call `handle_enrollment_recording()` during both voice and silence
3. Handle completion/abort status
4. Clear transcript after processing to avoid re-use

---

## 5. Client-Side Flag (Optional Enhancement)

Add to your JavaScript/TypeScript client:

```javascript
// Client state
let enrollmentRecordingActive = false;

// Handle enrollment start
websocket.on('message', (data) => {
    const message = JSON.parse(data);
    
    if (message.type === 'enrollment_prompt') {
        enrollmentRecordingActive = true;
        showEnrollmentUI(message.text);  // Display pangram
    }
    
    if (message.type === 'enrollment_reminder') {
        showReminder(message.text);  // Show "finish prompt" reminder
    }
    
    if (message.type === 'enrollment_complete' || 
        message.type === 'enrollment_aborted') {
        enrollmentRecordingActive = false;
        hideEnrollmentUI();
    }
});
```

---

## 6. Complete Example Flow

### Happy Path (Success):

1. **User**: "My name is John Smith"
2. **Rasa**: Detects name → triggers enrollment → calls `action_start_enrollment_recording`
3. **Action**: Posts to `/api/record_pangram` → **returns immediately**
4. **Server**: `record_pangram()` sets up state → sends pangram text to client → returns `{'status': 'started'}`
5. **Rasa**: Sets `enrollment_active=True` → continues normal operation
6. **Client**: Shows pangram, user starts reciting
7. **Server Loop**: Continuously processes audio in `process_audio_from_queue()`
   - Collects audio chunks in memory buffer
   - Monitors transcripts for completion
   - Checks ECAPA for other speakers
   - Checks for timeouts
8. **User**: Completes pangram (90% match detected)
9. **Server**: `complete_enrollment()` saves WAV → updates imprint → sends success message
10. **Server**: Sends `"SYSTEM_ENROLLMENT_COMPLETE_SUCCESS"` to Rasa webhook
11. **Rasa**: Receives system message → triggers rule → resets `enrollment_active=False`
12. **Done**: User can continue normal conversation

### Abort Path (Other Speaker):

1-6. Same as above
7. **Server Loop**: Detects different speaker with high confidence
8. **Server**: `abort_enrollment()` saves WAV → sends abort message
9. **Server**: Sends `"SYSTEM_ENROLLMENT_COMPLETE_ABORTED"` to Rasa
10. **Rasa**: Resets `enrollment_active=False`

---

## 7. Testing Checklist

### Basic Functionality:
- [ ] Recording starts when action triggered
- [ ] Pangram text sent to client (no TTS)
- [ ] Audio collected in memory buffer
- [ ] Final transcript processed correctly
- [ ] Fuzzy matching works (90% threshold)
- [ ] WAV file saved on completion (16-bit, 16kHz, mono)
- [ ] Speaker imprint created/updated
- [ ] Pangram marked as recited
- [ ] Rasa notified of completion

### Abort Conditions:
- [ ] Other speaker detected → abort
- [ ] 3 off-topic utterances → abort
- [ ] 7 seconds silence → abort
- [ ] "Cancel imprint" command → abort
- [ ] WAV saved on abort (for debugging)

### Edge Cases:
- [ ] User pauses mid-recitation → reminder sent
- [ ] All pangrams recited → random selection works
- [ ] New speaker (uid=None) → creates new entry
- [ ] Existing speaker → updates existing entry
- [ ] Empty audio buffer handled gracefully

### State Management:
- [ ] `enrollment_active` set to True when started
- [ ] `enrollment_active` reset to False on completion
- [ ] System messages trigger correct Rasa rules
- [ ] Client queue state cleaned up properly

---

## 8. Advantages of This Architecture

### vs Original Design:

| Aspect | Original Design | Simplified Design |
|--------|----------------|-------------------|
| **Complexity** | High (separate recording process) | Low (reuses existing loop) |
| **Blocking** | Yes (60s timeout) | No (returns immediately) |
| **State Sync** | Can desync on timeout | Always in sync |
| **Helper Functions** | Many new functions needed | Reuses existing infrastructure |
| **Audio Writing** | Live streaming to disk | In-memory → write on completion |
| **Data Access** | Special getters needed | Direct access in main loop |
| **Testing** | Complex (multiple processes) | Simpler (single flow) |
| **Maintenance** | High (more moving parts) | Low (fewer components) |

### Benefits:
1. **Simpler**: Uses existing audio processing infrastructure
2. **Faster**: No need to wait or poll for completion
3. **More Reliable**: No timeout issues or state desyncs
4. **Easier to Debug**: All logic in one place (main loop)
5. **More Efficient**: In-memory buffering avoids disk I/O during recording
6. **Better UX**: Rasa can respond to user immediately after starting

---

## 9. Next Steps

1. ✅ Copy `simplified_enrollment_implementation.py` functions to server01e.py
2. ✅ Update FastAPI endpoint `/api/record_pangram`
3. ✅ Update `ActionStartEnrollmentRecording` in actions.py
4. ✅ Add system intents and rules to Rasa (nlu.yml, domain.yml, rules.yml)
5. ✅ Add `ActionResetEnrollmentFlags` to actions.py
6. ✅ Integrate `handle_enrollment_recording()` into `process_audio_from_queue()`
7. ⚠️ Test thoroughly with various scenarios
8. ⚠️ Optional: Add client-side UI enhancements

---

## 10. Code Organization

### Suggested File Structure:

```
server01e.py
├── [imports]
├── CONFIG
├── setup_database()
├── [Model classes: PiperTTS, XTTSWrapper, etc.]
├── 
├── # === ENROLLMENT RECORDING FUNCTIONS ===
├── async def record_pangram()
├── async def handle_enrollment_recording()
├── async def complete_enrollment()
├── async def abort_enrollment()
├── async def save_audio_buffer_to_wav()
├── async def notify_rasa_enrollment_complete()
├── async def select_pangram_for_speaker()
├── async def mark_pangram_as_recited()
├── def normalize_text_for_comparison()
├── def calculate_fuzzy_match()
├── def is_utterance_on_topic()
├── def is_cancel_command()
├── # === END ENROLLMENT FUNCTIONS ===
├── 
├── async def process_audio_from_queue()  # [MODIFIED: add enrollment checks]
├── async def websocket_server()
├── async def connection_handler()
├── async def main()
```

### Or Separate Module:

You could also put enrollment functions in a separate file:

```python
# enrollment_recording.py
# [all enrollment functions]

# server01e.py
from enrollment_recording import (
    record_pangram,
    handle_enrollment_recording,
    complete_enrollment,
    abort_enrollment,
    # ... etc
)
```

This keeps server01e.py cleaner and enrollment logic modular.

---

## Summary

Your simplified architecture is **significantly better**:
- ✅ Non-blocking (action returns immediately)
- ✅ Uses existing infrastructure (no wheel reinvention)
- ✅ Simpler state management (everything in client_queues)
- ✅ In-memory buffering (write only on completion)
- ✅ Natural integration with existing audio loop
- ✅ Proper completion notification to Rasa

This design will be much easier to maintain and debug!
