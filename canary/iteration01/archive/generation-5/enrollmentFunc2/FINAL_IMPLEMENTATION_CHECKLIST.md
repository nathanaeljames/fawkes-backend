# Enrollment Recording - Final Implementation Checklist

## Quick Answers to Your Questions

### 1. ✅ Don't send to Rasa during enrollment (server-side check)
### 2. ✅ Timeout = simple timestamp comparison in loop
### 3. ✅ handle_enrollment_recording() is completely non-blocking
### 4. ✅ Use trigger_intent (cleaner than system message)

---

## Complete Implementation Steps

### Phase 1: Database Setup

**Run once:**
```bash
python3 database_updates.py
```

**Update server01e.py:**
- Replace `setup_database()` function with new version (includes pangrams table)

**Files:**
- `database_updates.py` - One-time migration script
- `updated_setup_database.py` - New setup function

---

### Phase 2: Server-Side Changes (server01e.py)

#### A. Add Enrollment Functions

Copy these functions from `simplified_enrollment_implementation.py`:

```python
# Add these functions to server01e.py:
- async def record_pangram()
- async def handle_enrollment_recording()
- async def complete_enrollment()
- async def abort_enrollment()
- async def save_audio_buffer_to_wav()
- async def notify_rasa_enrollment_complete()
- async def select_pangram_for_speaker()
- async def mark_pangram_as_recited()
- def normalize_text_for_comparison()
- def calculate_fuzzy_match()
- def is_utterance_on_topic()
- def is_cancel_command()
```

#### B. Update FastAPI Endpoint

```python
@app.post("/api/record_pangram")
async def record_pangram_endpoint(request: dict):
    """Kick off enrollment recording (returns immediately)"""
    client_id = request.get("client_id")
    uid = request.get("uid")
    firstname = request.get("firstname")
    surname = request.get("surname")
    
    if not client_id:
        raise HTTPException(status_code=400, detail="client_id required")
    
    result = await record_pangram(
        client_id=client_id,
        uid=uid,
        firstname=firstname,
        surname=surname
    )
    
    return result  # Returns {'status': 'started'} immediately
```

#### C. Modify process_audio_from_queue()

Add these checks:

```python
async def process_audio_from_queue(client_id, nemo_transcriber, nemo_vad, canary_qwen_transcriber):
    # ... existing initialization ...
    
    FINAL_TRANSCRIPT = ""
    
    while client_id in client_queues:
        try:
            audio_data = await asyncio.wait_for(
                client_queues[client_id]["incoming_audio"].get(), 
                timeout=1.0
            )
            
            # ... existing audio processing ...
            
            if is_voice_active_in_chunk:
                # ... existing voice processing ...
                
                # ===== NEW: CHECK FOR ENROLLMENT =====
                if "enrollment_state" in client_queues[client_id]:
                    if client_queues[client_id]["enrollment_state"]["recording_active"]:
                        result = await handle_enrollment_recording(
                            client_id=client_id,
                            audio_chunk=audio_int16,
                            is_voice_active=True,
                            current_transcript=FINAL_TRANSCRIPT,
                            speaker_name=SPEAKER_NAME,
                            speaker_uid=SPEAKER_UID,
                            speaker_confidence=SPEAKER_CONFIDENCE,
                            ecapa_processor=ecapa_processor
                        )
                        
                        if result in ['success', 'aborted']:
                            FINAL_TRANSCRIPT = ""
                            continue
                
                # ===== NEW: SKIP RASA DURING ENROLLMENT =====
                if FINAL_TRANSCRIPT:
                    if "enrollment_state" in client_queues[client_id]:
                        if client_queues[client_id]["enrollment_state"]["recording_active"]:
                            print("[Enrollment] Skipping Rasa - enrollment active")
                        else:
                            await send_to_rasa(client_id, FINAL_TRANSCRIPT)
                    else:
                        await send_to_rasa(client_id, FINAL_TRANSCRIPT)
                
                FINAL_TRANSCRIPT = ""
            
            else:  # Silence
                # ... existing silence handling ...
                
                # ===== NEW: CHECK ENROLLMENT DURING SILENCE =====
                if "enrollment_state" in client_queues[client_id]:
                    if client_queues[client_id]["enrollment_state"]["recording_active"]:
                        result = await handle_enrollment_recording(
                            client_id=client_id,
                            audio_chunk=np.array([], dtype=np.int16),
                            is_voice_active=False,
                            current_transcript="",
                            speaker_name=SPEAKER_NAME,
                            speaker_uid=SPEAKER_UID,
                            speaker_confidence=SPEAKER_CONFIDENCE,
                            ecapa_processor=ecapa_processor
                        )
                        
                        if result in ['success', 'aborted']:
                            continue
            
            # ... rest of existing processing ...
            
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.01)
```

**Key additions:**
1. Check for enrollment state
2. Call `handle_enrollment_recording()` during both voice and silence
3. Skip sending to Rasa when enrollment active
4. Handle completion/abort status

---

### Phase 3: Rasa-Side Changes

#### A. Update domain.yml

```yaml
# Add new intents (NO NLU examples needed!)
intents:
  # ... existing intents ...
  - system_enrollment_complete_success
  - system_enrollment_complete_aborted
```

#### B. Update rules.yml

```yaml
# Add rules for enrollment completion
- rule: Reset enrollment on success
  steps:
  - intent: system_enrollment_complete_success
  - action: action_reset_enrollment_flags

- rule: Reset enrollment on abort
  steps:
  - intent: system_enrollment_complete_aborted
  - action: action_reset_enrollment_flags
```

#### C. Update actions.py

**1. Modify ActionStartEnrollmentRecording:**

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
                    timeout=5  # Short timeout, just to start
                ) as response:
                    
                    if response.status == HTTPStatus.OK:
                        result = await response.json()
                        logger.info(f"Recording started: {result}")
                        
                        # Keep enrollment_active True
                        # Will be reset via trigger_intent
                        return [SlotSet("enrollment_active", True)]
                    else:
                        logger.error(f"Failed to start recording: {response.status}")
                        return [SlotSet("enrollment_active", False)]
                        
        except Exception as e:
            logger.error(f"Error starting enrollment: {e}")
            return [SlotSet("enrollment_active", False)]
```

**2. Add ActionResetEnrollmentFlags:**

```python
class ActionResetEnrollmentFlags(Action):
    """Reset enrollment flags when recording completes"""
    
    def name(self) -> Text:
        return "action_reset_enrollment_flags"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        intent_name = tracker.latest_message.get('intent', {}).get('name', '')
        
        if 'success' in intent_name:
            logger.info("Enrollment completed successfully - resetting flags")
        elif 'aborted' in intent_name:
            logger.info("Enrollment aborted - resetting flags")
        
        return [
            SlotSet("enrollment_active", False),
            SlotSet("enrollment_muted", False)
        ]
```

**3. Update domain.yml actions list:**

```yaml
actions:
  # ... existing actions ...
  - action_reset_enrollment_flags
```

---

### Phase 4: Optional Improvements

#### A. Client-Side UI (Optional)

```javascript
let enrollmentRecordingActive = false;

websocket.on('message', (data) => {
    const message = JSON.parse(data);
    
    if (message.type === 'enrollment_prompt') {
        enrollmentRecordingActive = true;
        showEnrollmentUI(message.text);
    }
    
    if (message.type === 'enrollment_reminder') {
        showReminder(message.text);
    }
    
    if (message.type === 'enrollment_complete' || 
        message.type === 'enrollment_aborted') {
        enrollmentRecordingActive = false;
        hideEnrollmentUI();
    }
});
```

#### B. Speaker Name Format Change (Optional)

If you want to change from underscore to space format:

**server01e.py** (4 locations):
```python
# Change from:
speaker_name = f"{firstname}_{surname}" if surname else firstname

# To:
speaker_name = f"{firstname} {surname}" if surname else firstname
```

**actions.py** (2 locations):
```python
# Line 80 - change from:
parts = ecapa_name.split("_", 1)

# To:
parts = ecapa_name.split(" ", 1)

# Line 696 - change from:
imprint_name = f"{query_firstname}_{query_surname}"

# To:
imprint_name = f"{query_firstname} {query_surname}"
```

---

## Testing Checklist

### Basic Functionality
- [ ] Database setup runs successfully
- [ ] Pangrams table created with 3 pangrams
- [ ] Recording starts when action triggered
- [ ] Pangram text sent to client
- [ ] Audio collected in memory buffer
- [ ] Rasa messages skipped during enrollment
- [ ] Final transcript processed correctly
- [ ] Fuzzy matching detects completion (90%)
- [ ] WAV file saved with correct filename
- [ ] Speaker imprint created/updated
- [ ] Pangram marked as recited
- [ ] Rasa notified via trigger_intent
- [ ] enrollment_active reset to False

### Abort Conditions
- [ ] Other speaker detected → abort
- [ ] 3 off-topic utterances → abort
- [ ] 7 seconds silence → abort
- [ ] "Cancel imprint" command → abort
- [ ] WAV saved on abort (debugging)

### Edge Cases
- [ ] User pauses mid-recitation → reminder sent
- [ ] All pangrams recited → random selection
- [ ] New speaker (uid=None) → creates entry
- [ ] Existing speaker → updates entry
- [ ] Empty audio buffer handled gracefully

### State Management
- [ ] enrollment_active set to True on start
- [ ] enrollment_active reset to False on completion
- [ ] Rasa doesn't respond during enrollment
- [ ] Client queue state cleaned up properly

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    ENROLLMENT FLOW                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. User: "My name is John Smith"                          │
│     ↓                                                       │
│  2. Rasa: Triggers enrollment                              │
│     ↓                                                       │
│  3. action_start_enrollment_recording                      │
│     ↓ (POST /api/record_pangram)                          │
│  4. Server: record_pangram()                               │
│     - Selects pangram                                      │
│     - Sets enrollment_state flag                           │
│     - Sends pangram to client                              │
│     - Returns {'status': 'started'} ✓                     │
│     ↓                                                       │
│  5. Action returns to Rasa (enrollment_active=True)        │
│     ↓                                                       │
│  6. User starts reciting pangram                           │
│     ↓                                                       │
│  7. Main audio loop: process_audio_from_queue()           │
│     - Checks enrollment_state flag                         │
│     - Calls handle_enrollment_recording()                  │
│     - Collects audio in memory                             │
│     - Monitors transcript for completion                   │
│     - Checks ECAPA for other speakers                      │
│     - Checks timeout                                        │
│     - Skips sending to Rasa                                │
│     ↓                                                       │
│  8. Completion detected (90% fuzzy match)                  │
│     ↓                                                       │
│  9. complete_enrollment()                                   │
│     - Saves WAV file                                       │
│     - Updates speaker imprint                              │
│     - Marks pangram as recited                             │
│     - Sends success message to client                      │
│     ↓                                                       │
│  10. notify_rasa_enrollment_complete()                     │
│      - POST /conversations/{id}/trigger_intent             │
│      - Intent: system_enrollment_complete_success          │
│      ↓                                                      │
│  11. Rasa: Receives intent                                 │
│      - Triggers rule                                       │
│      - Calls action_reset_enrollment_flags                 │
│      - Sets enrollment_active=False                        │
│      ↓                                                      │
│  12. Normal conversation resumes ✓                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

1. ✅ **Non-Blocking**: Action returns immediately, no waiting
2. ✅ **Reuses Infrastructure**: Uses existing audio loop
3. ✅ **In-Memory Buffering**: Write to disk only on completion
4. ✅ **Server-Side Logic**: Enrollment handled by server, not Rasa
5. ✅ **Simple State**: One flag in client_queues
6. ✅ **Direct Notification**: trigger_intent for completion
7. ✅ **Clean Separation**: Enrollment vs normal conversation

---

## Files Provided

### Implementation:
- `simplified_enrollment_implementation.py` - All enrollment functions
- `database_updates.py` - One-time database migration
- `updated_setup_database.py` - New setup function

### Documentation:
- `QUICK_ANSWERS.md` - Concise answers to your 4 questions
- `SIMPLIFIED_INTEGRATION_GUIDE.md` - Complete integration guide
- `enrollment_rasa_handling.py` - Rasa interruption solution
- `timeout_logic_explanation.py` - How timeout works
- `non_blocking_explanation.py` - Non-blocking proof
- `system_message_vs_trigger_intent.py` - trigger_intent details

### This File:
- `FINAL_IMPLEMENTATION_CHECKLIST.md` - You are here!

---

## Next Steps

1. ✅ Run `database_updates.py` once
2. ✅ Add enrollment functions to server01e.py
3. ✅ Update FastAPI endpoint
4. ✅ Modify process_audio_from_queue()
5. ✅ Update Rasa configuration (domain.yml, rules.yml, actions.py)
6. ✅ Test thoroughly
7. ⚠️ Optional: Add client-side UI
8. ⚠️ Optional: Change underscore to space format

**You're ready to implement!** 🚀
