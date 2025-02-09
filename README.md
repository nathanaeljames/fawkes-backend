Basic Python backend for Fawkes chatbot

- [X] Set up dockerfile for fast environment build
- [X] Figure out remote git issues
- [X] Receive audio data via websocket
- [ ] Relay audio to Google Speech
- [ ] Send sentence-by-sentence transcription to client via websockets
- [ ] Send audio to client via websockets
- [ ] Relay audio live to Watson
- [ ] Send live transcription with probabilities to client via websockets
- [ ] Basic response framework (name/ date/ wikipedia)
- [ ] ChatGPT/ Deepseek integration
- [ ] Speaker recognition
- [ ] Multiple speaker memory
- [ ] Prompting and phonetic pangram collection
- [ ] Custom voice
- [ ] Live model retraining against prompts and pangrams
- [ ] Interruptability

If docker is shut down improperly it may be necessary to run `rm .git/index.lock` to restore git functionality.

Watson Credentials:
  "apikey": "REDACTED",
  "iam_apikey_description": "REDACTED",
  "iam_apikey_id": "REDACTED",
  "iam_apikey_name": "Auto-generated service credentials",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Manager",
  "iam_serviceid_crn": "REDACTED",
  "url": "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/REDACTED_IBM_STT_INSTANCE"