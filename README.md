Basic Python backend for Fawkes chatbot

- [X] Set up dockerfile for fast environment build
- [X] Figure out remote git issues
- [X] Receive audio data via websocket
- [ ] Relay audio to Google Speech
- [X] Send sentence-by-sentence transcription to client via websockets (as text)
- [X] Send JSON data to client, display as text
- [ ] ~~Stream audio to client via websockets, play it automatically~~
- [X] Relay audio live to Watson
- [X] Send live transcription with probabilities to client via websockets
- [X] Capture audio sample for quality check
- [ ] Resolve Docker network connectivity issues
- [ ] Docker network/ remote server test
- [X] Basic response framework (name/ date/ wikipedia)
- [ ] Implement own local STT model
- [ ] Implement own local TTS model
- [ ] ChatGPT/ Deepseek integration
- [ ] Speaker recognition
- [ ] Multiple speaker memory
- [ ] Prompting and phonetic pangram collection
- [ ] Custom voice
- [ ] Live model retraining against prompts and pangrams
- [ ] Interruptability

* ISSUE 1: If docker is shut down improperly it may be necessary to run `rm .git/index.lock` to restore git functionality.
* ISSUE 2: Docker "rebuild" insists on using cache, instantly reusing broken container and ignoring any revisions to Dockerfile, must run `docker system prune -a`
