# Create the Rasa project structure in your ./rasa directory
mkdir -p ./rasa/data
mkdir -p ./rasa/actions

# Create domain.yml in ./rasa/
cat > ./rasa/domain.yml << 'EOF'
version: "3.1"

intents:
  - greet
  - goodbye
  - ask_time
  - ask_name
  - affirm
  - deny

entities: []

slots: {}

responses:
  utter_greet:
  - text: "Hello! How can I help you?"

  utter_goodbye:
  - text: "Goodbye! Have a great day!"

  utter_ask_name:
  - text: "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."

  utter_default:
  - text: "I'm sorry, I didn't understand that. Could you please rephrase?"

actions:
  - action_tell_time

session_config:
  session_expiration_time: 60
  carry_over_slots_to_new_session: true
EOF

# Create nlu.yml in ./rasa/data/
cat > ./rasa/data/nlu.yml << 'EOF'
version: "3.1"

nlu:
- intent: greet
  examples: |
    - hey
    - hello
    - hi
    - hello there
    - good morning
    - good evening
    - hey there

- intent: goodbye
  examples: |
    - cu
    - good by
    - cee you later
    - good night
    - bye
    - goodbye
    - have a nice day
    - see you around
    - bye bye
    - see you later

- intent: ask_time
  examples: |
    - what time is it
    - tell me the time
    - what's the time
    - current time
    - time please
    - the time
    - what time

- intent: ask_name
  examples: |
    - what's your name
    - tell me your name
    - who are you
    - your name
    - what are you called
EOF

# Create stories.yml in ./rasa/data/
cat > ./rasa/data/stories.yml << 'EOF'
version: "3.1"

stories:

- story: happy path
  steps:
  - intent: greet
  - action: utter_greet

- story: ask time
  steps:
  - intent: ask_time
  - action: action_tell_time

- story: ask name
  steps:
  - intent: ask_name
  - action: utter_ask_name

- story: say goodbye
  steps:
  - intent: goodbye
  - action: utter_goodbye
EOF

# Create rules.yml in ./rasa/data/
cat > ./rasa/data/rules.yml << 'EOF'
version: "3.1"

rules:

- rule: Say goodbye anytime the user says goodbye
  steps:
  - intent: goodbye
  - action: utter_goodbye

- rule: Tell time when asked
  steps:
  - intent: ask_time
  - action: action_tell_time
EOF

# Create config.yml in ./rasa/
cat > ./rasa/config.yml << 'EOF'
recipe: default.v1

language: en

pipeline:
- name: WhitespaceTokenizer
- name: RegexFeaturizer
- name: LexicalSyntacticFeaturizer
- name: CountVectorsFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: DIETClassifier
  epochs: 100
  constrain_similarities: true
- name: EntitySynonymMapper
- name: ResponseSelector
  epochs: 100
  constrain_similarities: true
- name: FallbackClassifier
  threshold: 0.3
  ambiguity_threshold: 0.1

policies:
- name: MemoizationPolicy
- name: RulePolicy
- name: UnexpecTEDIntentPolicy
  max_history: 5
  epochs: 100
- name: TEDPolicy
  max_history: 5
  epochs: 100
  constrain_similarities: true
EOF

# Create actions.py in ./rasa/actions/
cat > ./rasa/actions/actions.py << 'EOF'
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import datetime

class ActionTellTime(Action):

    def name(self) -> Text:
        return "action_tell_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        message = f"Sir, the time is {current_time}"
        
        dispatcher.utter_message(text=message)
        
        return []
EOF

# Create endpoints.yml in ./rasa/
cat > ./rasa/endpoints.yml << 'EOF'
action_endpoint:
  url: "http://localhost:5055/webhook"
EOF

# Create credentials.yml in ./rasa/
cat > ./rasa/credentials.yml << 'EOF'
rest:
  # you don't need to provide anything here - this channel doesn't
  # require any credentials
EOF

echo "Rasa project structure created successfully!"
echo "Next steps:"
echo "1. Run: docker-compose up -d"
echo "2. Train the Rasa model: docker-compose exec rasa rasa train"
echo "3. Update your CONFIG in server01e.py to use 'http://rasa:5005'"
