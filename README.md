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
- [X] Robust VAD for finality determination
- [ ] Speaker diarization & segmentation
- [X] Speaker recognition
- [X] Multiple speaker memory using duckdb for embeddings
- [X] Incremental ECAPA embeddings using high-scoring utterances
- [ ] Speaker embeddings/ adapter layer to improve ASR accuracy
- [ ] Live model retraining against prompts and pangrams/ unsupervised model(?)
- [ ] ~~Add a context-aware language model rescoring step (e.g., GPT, BERT) during final result compilation~~
- [ ] "Skills framework" (Rasa?)
- [ ] Prompting and phonetic pangram collection
- [ ] ~~ChatGPT/ Deepseek integration~~
- [ ] Langchain routing and LLM referal
- [ ] RAG use case
- [ ] Interruptability
- [ ] Model serving framework for multiclient support using stateful models

* ISSUE 1: If docker is shut down improperly it may be necessary to run `rm .git/index.lock` to restore git functionality.
* ISSUE 2: Docker "rebuild" insists on using cache, instantly reusing broken container and ignoring any revisions to Dockerfile, must run `docker system prune -a`
* NOTE: Use command `watch -n 1 nvidia-smi` to track GPU memory usage live

NOTES:
* Currently using fastconformer hybrid cache-aware streaming transformer<sup>[1](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi),[2](https://arxiv.org/abs/2312.17279)</sup>, which is state-of-the-art for (streaming) transformer-based ASR. However, may eventually replace this with Mamba state-space models which are showing much better promise<sup>[3](https://arxiv.org/abs/2407.09732)</sup> for long-context STT, and especially speech seperation<sup>[4](https://arxiv.org/html/2410.06459v2),[5](https://arxiv.org/abs/2403.18257)</sup> and diarization<sup>[6](https://www.researchgate.net/publication/384770025_Mamba-based_Segmentation_Model_for_Speaker_Diarization)</sup>. SSM models are more in the research stage and may require some engineering for streaming audio support.
* **First pass:** set up VAD, basic speaker recognition, transformer-based STT, context-aware LM rescoring. VAD is usefull regardles of future pipelines, can give a "soft" end-of-utterance for batch processing of N-best candidates/beam search by contextual LM. VAD finality can be second guessed by LM rescoring. LM rescoring will be useful whether using transformers or SSM in the future as only LM can provide full grammatical contextual awareness. Speaker recognition will function in the same way regardless of future speech seperation development. Direct instantiation for multiclient support.
* **Modified first pass** the traditional Nemo ecosystem would call for me at this point (using fastconformer ASR) to set up a context-aware LM rescoring model (had settled on pretrained [transformer-xl](https://huggingface.co/transfo-xl/transfo-xl-wt103/tree/main)) to rescore beam search, then Inverse Text Normalization (ITN), then Punctuation and Capitalization (P&C) (using separate Nemo models for each). Ultimately I would then circle back to set up [Language Model Fusion](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_language_modeling_and_customization.html) and a framework to support it like NGPU-LM. However, all these steps may be rendered obsolete as the apogee of the Nemo framework, [Canary-Qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) already uses a full-fledged large language model (Qwen) with full decoder fusion to top the HF ASR leaderboard with 5.6% WER (true state-of-the-art for transformers). So making a major pivot to use this new model for final inference on the utterance level, may scale back interim ASR model to citrinet in the future (more light weight and snappy) but proceeding w/ Fastconformer and Canary-Qwen-2.5b for now. SSM models are reporting ~3% WER even for ASR. I will defer experimental SSM adoption to the second pass.
* **Second pass:** Investigate SSM models, speech seperation (e.g. svoice Multi-stage Gated NN<sup>[1](https://github.com/facebookresearch/svoice)</sup> vs Mamba SSM<sup>[2](https://github.com/xi-j/Mamba-TasNet)</sup>), CASA/dereverberation<sup>[3](https://pmc.ncbi.nlm.nih.gov/articles/PMC7473777/)</sup>, possible replacement of transformer ASR with SSM ASR. Set up inference server for large scale dynamic batching and adapter-layer management.
