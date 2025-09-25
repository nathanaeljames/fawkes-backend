from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import datetime
from rasa_sdk.events import SlotSet


class ActionSetFirstname(Action):
    def name(self):
        return "action_set_firstname"

    def run(self, dispatcher, tracker, domain):
        speaker_name = tracker.get_slot("speaker_name")
        
        if speaker_name and speaker_name != "unknown speaker":
            firstname = speaker_name.split("_")[0].capitalize()
        else:
            firstname = None
        
        return [SlotSet("firstname", firstname)]

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
    
class ActionSetPartOfDay(Action):
    def name(self):
        return "action_set_part_of_day"

    def run(self, dispatcher, tracker, domain):
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            part_of_day = "morning"
        elif 12 <= hour < 17:
            part_of_day = "afternoon" 
        elif 17 <= hour < 21:
            part_of_day = "evening"
        else:
            part_of_day = "evening"
        
        return [SlotSet("part_of_day", part_of_day)]

class ActionDoNothing(Action):
    def name(self):
        return "action_do_nothing"

    def run(self, dispatcher, tracker, domain):
        # This action does nothing and returns an empty list of events
        return []