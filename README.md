Basic Python backend for Fawkes chatbot

- [X] Set up dockerfile for fast environment build
- [X] Figure out remote git issues
- [X] Receive audio data via websocket
- [ ] ~~Relay audio to Google Speech~~
- [X] Send sentence-by-sentence transcription to client via websockets (as text)
- [X] Send JSON data to client, display as text
- [X] Stream audio to client via websockets, play it automatically
- [X] Relay audio live to Watson
- [X] Send live transcription with probabilities to client via websockets
- [X] Capture audio sample for quality check
- [X] Resolve Docker network connectivity issues
- [X] Docker network/ remote server test
- [X] Basic response framework (name/ date/ wikipedia)
- [X] Implement own local TTS model
- [X] Zero shot voice cloning functionality
- [X] Voice cloning from speaker embeds/ gpt latents (as .pt files)
- [X] Custom voice
- [X] Multiclient support, asyncronous model/routine calls
- [ ] Implement own local STT model
- [ ] Add a context-aware language model rescoring step (e.g., GPT, BERT) during final result compilation
- [ ] Live model retraining against prompts and pangrams/ unsupervised model(?)
- [ ] Speaker embeddings/ adapter layer to improve ASR accuracy
- [ ] Speaker recognition
- [ ] Multiple speaker memory
- [ ] "Skills framework" (Rasa?)
- [ ] Prompting and phonetic pangram collection
- [ ] ~~ChatGPT/ Deepseek integration~~
- [ ] Langchain routing and LLM referal
- [ ] RAG use case
- [ ] Interruptability

* ISSUE 1: If docker is shut down improperly it may be necessary to run `rm .git/index.lock` to restore git functionality.
* ISSUE 2: Docker "rebuild" insists on using cache, instantly reusing broken container and ignoring any revisions to Dockerfile, must run `docker system prune -a`
