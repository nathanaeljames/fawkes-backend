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

* ISSUE 1: If docker is shut down improperly it may be necessary to run `rm .git/index.lock` to restore git functionality.
* ISSUE 2: Docker "rebuild" insists on using cache, instantly reusing broken container and ignoring any revisions to Dockerfile, must run `docker system prune -a`

Watson Credentials:
  "apikey": "IYBIxRJeINqwcjOAd0PuFYI6NLyH0qV8hqfh3ziNqtQf",
  "iam_apikey_description": "Auto-generated for key crn:v1:bluemix:public:speech-to-text:us-south:a/1a669a2356c6484c969d898d5438de4e:30d589a2-77a6-4819-90f7-9a3090278b40:resource-key:84a2cf88-3165-43e6-82ec-1e1118d1bc3c",
  "iam_apikey_id": "ApiKey-8c279fd8-dbcb-436a-9734-2e33156e3879",
  "iam_apikey_name": "Auto-generated service credentials",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Manager",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/1a669a2356c6484c969d898d5438de4e::serviceid:ServiceId-1ab9330c-12e4-4715-807b-bbbb30b2bc5b",
  "url": "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/30d589a2-77a6-4819-90f7-9a3090278b40"