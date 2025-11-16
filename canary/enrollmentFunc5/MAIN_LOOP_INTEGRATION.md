# MAIN LOOP INTEGRATION - CORRECTED VERSION

## Quick Integration Snippet

Here's exactly what to add to your `process_audio_from_queue()` function:

```python
async def process_audio_from_queue(client_id, nemo_transcriber, nemo_vad, canary_qwen_transcriber):
    """Main audio processing loop with enrollment recording integration"""
    
    # ... existing setup code ...
    
    # === NEW: CLIENT-LEVEL timestamp (NOT in enrollment_state) ===
    last_utterance_time = asyncio.get_event_loop().time()
    
    current_utterance_buffer = []
    is_speaking = False
    silence_counter = 0
    SILENCE_CHUNKS_THRESHOLD = 10  # Your value
    
    while client_id in client_queues:
        try:
            audio_chunk = await client_queues[client_id]["incoming_audio"].get()
            current_time = asyncio.get_event_loop().time()
            
            # ... your existing VAD processing ...
            # voice_active = ...
            
            # ============================================================
            # VOICE ACTIVE SECTION
            # ============================================================
            if voice_active:
                if not is_speaking:
                    is_speaking = True
                    silence_counter = 0
                    print("Voice activity started")
                
                # Just accumulate audio during voice active
                current_utterance_buffer.append(audio_chunk)
                
                # NOTE: NO enrollment processing here!
                # We process utterance-by-utterance, not chunk-by-chunk
            
            # ============================================================
            # SILENCE SECTION
            # ============================================================
            else:
                silence_counter += 1
                
                # ========================================================
                # ENROLLMENT: DIRECT TIMEOUT & REMINDER CHECKS
                # ========================================================
                if "enrollment_state" in client_queues[client_id]:
                    enrollment_state = client_queues[client_id]["enrollment_state"]
                    
                    if enrollment_state["recording_active"]:
                        silence_duration = current_time - last_utterance_time
                        
                        # --- TIMEOUT CHECK (runs every silence chunk) ---
                        if silence_duration >= CONFIG['enrollment_timeout']:
                            print(f"[Enrollment] Timeout after {silence_duration:.1f}s")
                            await enrollment_recording_manager.abort_recording(
                                client_id,
                                reason="timeout"
                            )
                        
                        # --- REMINDER CHECK (runs every silence chunk) ---
                        elif silence_duration >= CONFIG['enrollment_reminder_interval']:
                            last_prompt_time = enrollment_state.get("last_prompt_time")
                            
                            # Only send if we haven't sent one recently
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
                
                # ========================================================
                # PROCESS UTTERANCE FINALITY
                # ========================================================
                if is_speaking and silence_counter >= SILENCE_CHUNKS_THRESHOLD:
                    print("Acoustic finality detected. Processing full utterance...")
                    
                    # Update CLIENT-LEVEL timestamp
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
                            print(f"[Final Speaker ID] {final_speaker_match}")
                    
                    # Get final transcript (your existing Canary-Qwen code)
                    # Uncomment when ready to use Canary-Qwen:
                    # final_transcription_text = await canary_qwen_transcriber.transcribe(...)
                    final_transcription_text = ""  # TODO: Replace with actual transcript
                    
                    # ====================================================
                    # ENROLLMENT: PROCESS UTTERANCE
                    # ====================================================
                    enrollment_handled = False
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
                                print(f"[Enrollment] Recording {result}")
                            
                            enrollment_handled = True
                    
                    # ====================================================
                    # NORMAL RASA PROCESSING (skip if enrollment active)
                    # ====================================================
                    if not enrollment_handled:
                        # Your existing Rasa code
                        # ... send to Rasa ...
                        pass
                    else:
                        print("[Enrollment] Skipped Rasa (enrollment active)")
                    
                    # Reset for next utterance
                    current_utterance_buffer.clear()
                    is_speaking = False
        
        except Exception as e:
            print(f"Error in audio processing: {e}")
            continue
```

---

## Key Points

### 1. Client-Level Timestamp
```python
# Initialize ONCE per client (outside main loop if persisting)
last_utterance_time = asyncio.get_event_loop().time()

# Update when utterance completes
if is_speaking and silence_counter >= SILENCE_CHUNKS_THRESHOLD:
    last_utterance_time = current_time
```

**Why client-level?**
- Can be used for other purposes (not just enrollment)
- Tracks when user last spoke (general audio processing info)
- Simpler architecture

---

### 2. Direct Timeout Check
```python
# NO method call - just check directly:
if silence_duration >= CONFIG['enrollment_timeout']:
    await enrollment_recording_manager.abort_recording(
        client_id,
        reason="timeout"
    )
```

**Why direct?**
- Simpler
- No unnecessary method call
- Clear what's happening
- Easier to debug

---

### 3. Direct Reminder Logic
```python
# NO method call - just send directly:
elif silence_duration >= CONFIG['enrollment_reminder_interval']:
    last_prompt_time = enrollment_state.get("last_prompt_time")
    
    if last_prompt_time is None or (current_time - last_prompt_time) >= CONFIG['enrollment_reminder_interval']:
        # Send reminder
        message = {"type": "enrollment_reminder", "message": reminder_text}
        await send_message_to_client(client_id, json.dumps(message))
        enrollment_state["last_prompt_time"] = current_time
```

**Why direct?**
- Clear timing logic
- No hidden state checks
- Easier to modify intervals
- More maintainable

---

### 4. Public abort_recording() with Reasons
```python
# Call with reason code:
await enrollment_recording_manager.abort_recording(
    client_id,
    reason="timeout"  # or "other_speaker", "cancel", "off_topic"
)
```

**Why reason codes?**
- Consistent abort messages
- Better logging/debugging
- Easy to add new reasons
- Clear API

---

## Configuration

Add these to your CONFIG dict:

```python
CONFIG = {
    # ... existing config ...
    
    # Enrollment settings
    'enrollment_timeout': 7.0,              # Seconds before timeout
    'enrollment_reminder_interval': 2.0,    # Seconds between reminders
    'enrollment_min_match': 0.90,           # 90% match to complete
    'enrollment_max_off_topic': 3,          # Max off-topic utterances
    'samples_path': '/path/to/samples',     # Where to save WAV files
    'rasa_url': 'http://localhost:5005',    # Rasa server
    'clientSideTTS': False,                 # Client handles TTS?
}
```

---

## Initialization

In your `main()` function:

```python
async def main():
    # ... existing setup ...
    
    # Initialize enrollment recording manager
    enrollment_recording_manager = EnrollmentRecordingManager(
        db_connection=con,
        config=CONFIG,
        client_queues=client_queues
    )
    
    # Initialize API handler
    enrollment_api_handler = EnrollmentAPIHandler(
        db_connection=con,
        recording_manager=enrollment_recording_manager
    )
    
    # ... rest of setup ...
```

---

## FastAPI Endpoint

```python
@app.post("/api/record_pangram")
async def record_pangram_endpoint(request: dict):
    """Start enrollment recording (returns immediately)"""
    return await enrollment_api_handler.record_pangram(request)
```

---

## What Gets Called When

### Timeline of a Typical Enrollment:

```
t=0s:   Rasa calls /api/record_pangram
        → enrollment_recording_manager.start_recording()
        → Initializes enrollment_state
        → Sends initial prompt
        → Returns immediately

t=1s:   User starts speaking
        → Voice active section: accumulate audio
        → No enrollment processing yet

t=3s:   User stops speaking
        → Silence section: silence_counter++
        → Check timeout (3s < 7s, OK)
        → Check reminder (3s > 2s, send reminder)

t=4s:   Silence continues
        → silence_counter >= threshold
        → Process utterance:
           • Get ECAPA result
           • Get Canary-Qwen transcript
           • Call process_utterance()
           • Check fuzzy match
           • Still recording...
        → Update last_utterance_time = 4s

t=5s:   Silence continues
        → Check timeout (1s since last utterance, OK)
        
t=6s:   Silence continues
        → Check reminder (2s since last utterance, send reminder)

t=7s:   User speaks again
        → Voice active: accumulate audio

t=9s:   User stops
        → Process utterance again
        → Match score >= 90%
        → _complete_recording()
        → Save WAV
        → Update DB
        → Notify Rasa
        → Cleanup state
        → DONE!
```

---

## Common Abort Scenarios

### Timeout:
```python
t=0s: Start recording
t=4s: User speaks (last_utterance_time = 4s)
t=11s: Still silent (11 - 4 = 7s)
      → abort_recording(reason="timeout")
```

### Other Speaker:
```python
t=4s: User speaks
      → ECAPA: speaker_match = "uid123"
      → process_utterance() detects mismatch
      → abort_recording(reason="other_speaker")
```

### Cancel:
```python
t=4s: User says "cancel"
      → process_utterance() detects cancel keyword
      → abort_recording(reason="cancel")
```

### Off-Topic:
```python
t=2s: User says something off-topic (count=1)
t=4s: User says something off-topic (count=2)
t=6s: User says something off-topic (count=3)
      → process_utterance() detects max reached
      → abort_recording(reason="off_topic")
```

---

## Quick Debug Checklist

If enrollment not working:

1. [ ] Check enrollment_recording_manager initialized in main()
2. [ ] Check /api/record_pangram endpoint hooked up
3. [ ] Check enrollment_state exists when expected
4. [ ] Check last_utterance_time updating on finality
5. [ ] Check timeout/reminder logic running in silence section
6. [ ] Check process_utterance() called with correct parameters
7. [ ] Check Rasa being skipped during enrollment
8. [ ] Check CONFIG values are reasonable

---

## That's It!

The corrected implementation is complete and ready to use. The main changes:
1. ✅ No check_timeout() method - do it inline
2. ✅ No send_reminder_if_needed() method - do it inline
3. ✅ Public abort_recording() with reason codes
4. ✅ Simpler, more maintainable architecture
