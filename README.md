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
- [X] Implement own local STT model
- [X] Fine-tune basline STT model
- [ ] Speaker diarization, segmentation, robust VAD for finality determination
- [ ] Speaker recognition
- [ ] Multiple speaker memory
- [ ] Speaker embeddings/ adapter layer to improve ASR accuracy
- [ ] Live model retraining against prompts and pangrams/ unsupervised model(?)
- [ ] Add a context-aware language model rescoring step (e.g., GPT, BERT) during final result compilation
- [ ] "Skills framework" (Rasa?)
- [ ] Prompting and phonetic pangram collection
- [ ] ~~ChatGPT/ Deepseek integration~~
- [ ] Langchain routing and LLM referal
- [ ] RAG use case
- [ ] Interruptability

* ISSUE 1: If docker is shut down improperly it may be necessary to run `rm .git/index.lock` to restore git functionality.
* ISSUE 2: Docker "rebuild" insists on using cache, instantly reusing broken container and ignoring any revisions to Dockerfile, must run `docker system prune -a`

NOTES:
* Currently using fastconformer hybrid cache-aware streaming transformer[1](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi), which is state-of-the-art for transformer-based streaming ASR. However, may eventually replace this with Mamba state-space models which are showing much better promise [2](https://arxiv.org/abs/2407.09732) for long-context STT, and especially speech seperation [3](https://arxiv.org/html/2410.06459v2) [4](https://arxiv.org/abs/2403.18257) and diarization [5](https://www.researchgate.net/publication/384770025_Mamba-based_Segmentation_Model_for_Speaker_Diarization). SSM models are more in the research stage and may require some engineering for streaming audio support.
* First pass: set up VAD, basic speaker recognition, transformer-based STT, context-aware LM rescoring.
* VAD is usefull regardles of pipeline, can give a "soft" end-of-utterance for batch processing of N-best candidates/beam search by contextual LM. VAD finality can be second guessed by LM rescoring.
* Second pass: Investigate SSM models, speech seperation (e.g. [svoice](https://github.com/facebookresearch/svoice) Multi-stage Gated NN) vs Mamba SSM [1](https://github.com/xi-j/Mamba-TasNet), CASA/dereverberation [2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7473777/), possible replacement of transformer ASR with SSM ASR
