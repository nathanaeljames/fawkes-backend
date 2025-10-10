from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
import datetime
from rasa_sdk.types import DomainDict
from rasa_sdk.events import SlotSet
import logging
import pytz
import aiohttp
from http import HTTPStatus

EST_TZ = pytz.timezone('America/New_York')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================
# GLOBAL CONFIGURATION FOR FASTAPI SERVER (server01e.py)
# NOTE: These values must match the binding configuration in server01e.py
FASTAPI_HOST = "0.0.0.0"
FASTAPI_PORT = 9002
# ==========================================================

class ActionLogSlots(Action):
    def name(self) -> Text:
        return "action_log_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get current slot values
        speaker_name = tracker.get_slot("speaker_name")
        firstname = tracker.get_slot("firstname") 
        imprint_name = tracker.get_slot("imprint_name")
        
        # Log to console with timestamp
        logger.info(f"SLOT VALUES - speaker_name: '{speaker_name}', firstname: '{firstname}', imprint_name: '{imprint_name}'")
        
        # Also log the latest message and sender for context
        latest_message = tracker.latest_message.get("text", "")
        sender = tracker.sender_id
        
        logger.info(f"MESSAGE CONTEXT - sender: {sender}, message: '{latest_message}'")
        
        return []

class ActionSetNameSlots(Action):
    def name(self) -> Text:
        return "action_set_name_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get speaker from metadata
        metadata = tracker.latest_message.get("metadata", {})
        speaker_name = metadata.get("speaker_name")

        EXCLUDED_SPEAKERS = {"unknown_speaker", "unknown speaker", "unregistered"}
        
        # Initialize slots
        firstname = None
        surname = None
        
        # Set speaker_name and extract firstname/surname if valid
        if speaker_name and speaker_name not in EXCLUDED_SPEAKERS:
            parts = speaker_name.split("_", 1)  # Split on first underscore only
            firstname = parts[0].capitalize()
            if len(parts) > 1:
                surname = parts[1].capitalize()
        
        return [
            SlotSet("speaker_name", speaker_name),
            SlotSet("firstname", firstname),
            SlotSet("surname", surname)
        ]

class ActionQueryUserbase(Action):
    def name(self) -> Text:
        return "action_query_userbase"
    
    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        firstname = tracker.get_slot("firstname")
        surname = tracker.get_slot("surname")
        
        if not firstname or not surname:
            logger.warning(f"Missing name information - firstname: {firstname}, surname: {surname}")
            return [SlotSet("imprint_uid", None)]
        
        # Query the FastAPI server to check if speaker exists in database
        fastapi_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/query"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    fastapi_url,
                    json={
                        "action": "query",
                        "table": "speakers",
                        "firstname": firstname,
                        "surname": surname
                    },
                    timeout=5
                ) as response:
                    
                    if response.status == HTTPStatus.OK:
                        result = await response.json()
                        uid = result.get("uid")
                        
                        if uid:
                            logger.info(f"Found speaker {firstname} {surname} with UID: {uid}")
                            imprint_name = f"{firstname}_{surname}" if surname else firstname
                            return [
                                SlotSet("imprint_uid", str(uid)),
                                SlotSet("imprint_name", imprint_name)
                            ]
                        else:
                            logger.info(f"No record found for {firstname} {surname}")
                            return [SlotSet("imprint_uid", None)]
                    else:
                        logger.error(f"Server returned status {response.status}")
                        return [SlotSet("imprint_uid", None)]
                        
        except Exception as e:
            logger.error(f"Error querying userbase: {e}")
            return [SlotSet("imprint_uid", None)]

class ActionStartEnrollmentRecording(Action):
    def name(self) -> Text:
        return "action_start_enrollment_recording"
    
    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        sender_id = tracker.sender_id
        imprint_uid = tracker.get_slot("imprint_uid")
        
        # Trigger recording on server
        fastapi_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/record"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    fastapi_url,
                    json={
                        "action": "record",
                        "uid": imprint_uid
                    },
                    timeout=5
                ) as response:
                    
                    if response.status == HTTPStatus.OK:
                        result = await response.json()
                        logger.info(f"Recording started for {sender_id}: {result}")
                        return [SlotSet("enrollment_active", True)]
                        # NOTE instead of trigger enrollment_active I will need to process a success message from the server and set enrollment_active to False rasa and server-side
                    else:
                        logger.error(f"Failed to start recording: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error starting enrollment recording: {e}")
            return []

class ActionMuteEnrollment(Action):
    def name(self) -> Text:
        return "action_mute_enrollment"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info(f"Muting enrollment suggestions for session")
        return [SlotSet("enrollment_muted", True)]

class ActionResetEnrollment(Action):
    def name(self) -> Text:
        return "action_reset_enrollment"
    
    async def run(self, dispatcher, tracker, domain):
        sender_id = tracker.sender_id
        fastapi_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/enrollment_status"

        logger.info(f"Attempting to reset enrollment for client {sender_id} via FastAPI endpoint: {fastapi_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    fastapi_url, 
                    json={"client_id": sender_id, "status": "abandoned"},
                    # Use a short timeout since this is an internal communication
                    timeout=5 
                ) as response:
                    
                    if response.status == HTTPStatus.OK:
                        logger.info(f"Enrollment reset request successful for {sender_id}. Server response: {await response.json()}")
                    else:
                        logger.warning(f"Server returned non-200 status ({response.status}) for reset request: {await response.text()}")

        except aiohttp.ClientConnectorError:
            # Handle case where the server is down or unreachable
            logger.error(f"Could not connect to FastAPI server at {fastapi_url}.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during API call: {e}")
        
        # Always set the slot, even if the API call failed, to ensure Rasa's state is updated
        return [SlotSet("enrollment_active", False)]

class ActionSetTimeOfDay(Action):

    def name(self) -> Text:
        return "action_set_time_of_day"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        #current_time = datetime.datetime.now().strftime("%H:%M:%S")
        current_time_aware = datetime.datetime.now(EST_TZ)
        current_time = current_time_aware.strftime("%H:%M:%S")
        #message = f"Sir, the time is {current_time}"
        
        return [SlotSet("time_of_day", current_time)]
    
class ActionSetPartOfDay(Action):
    def name(self):
        return "action_set_part_of_day"

    def run(self, dispatcher, tracker, domain):
        #hour = datetime.datetime.now().hour
        hour = datetime.datetime.now(EST_TZ).hour 
        
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

# Enrollment actions

class ActionProcessSpelling(Action):
    """Process hyphenated spelling input (e.g., N-A-T-E) and convert to word"""
    # TODO handle 'space', 'apostrophe', 'dash', 'hyphen', remove dependence on hyphenation
    def name(self) -> Text:
        return "action_process_spelling"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        text = tracker.latest_message.get("text", "").strip()
        intent = tracker.latest_message.get("intent", {}).get("name")
        spelling_stage = tracker.get_slot("spelling_stage")
        
        logger.info(f"Processing spelling input: '{text}' (intent: {intent}) at stage: {spelling_stage}")
        
        # Validate that we got the spell_name intent
        if intent != "spell_name":
            logger.warning(f"Expected spell_name intent but got {intent}")
            dispatcher.utter_message(text="Please spell your name letter-by-letter, like N-A-T-E.")
            return []
        
        # Check if input is in hyphenated format (e.g., "N-A-T-E")
        if "-" not in text:
            logger.warning(f"Spelling input missing hyphens: '{text}'")
            dispatcher.utter_message(text="Please spell your name with hyphens between letters, like N-A-T-E.")
            return []
        
        # Remove hyphens and spaces, convert to title case
        letters = text.replace("-", "").replace(" ", "").upper()
        
        # Validate that we have reasonable input (letters only, reasonable length)
        if not letters.isalpha() or len(letters) < 2 or len(letters) > 30:
            logger.warning(f"Invalid spelling format: '{text}'")
            dispatcher.utter_message(text="I didn't understand that spelling. Please try again.")
            return []
        
        processed_name = letters.capitalize()
        
        # Store the formatted spelling for confirmation
        formatted_spelling = "-".join(list(letters))
        
        logger.info(f"Converted spelling '{text}' to name '{processed_name}'")
        
        if spelling_stage == "spelling_first":
            #dispatcher.utter_message(response="utter_confirm_firstname_spelling")
            dispatcher.utter_message(text=f"{processed_name}, spelled {formatted_spelling}. Did I get that right?")
            return [
                SlotSet("imprint_firstname", processed_name),
                SlotSet("firstname_spelled", formatted_spelling),
                SlotSet("spelling_stage", "confirming_first")
            ]
        elif spelling_stage == "spelling_last":
            #dispatcher.utter_message(response="utter_confirm_surname_spelling")
            dispatcher.utter_message(text=f"{processed_name}, spelled {formatted_spelling}. Did I get that right?")
            return [
                SlotSet("imprint_surname", processed_name),
                SlotSet("surname_spelled", formatted_spelling),
                SlotSet("spelling_stage", "confirming_last")
            ]
        else:
            logger.error(f"Unexpected spelling_stage: {spelling_stage}")
            return []

class ActionConfirmFullName(Action):
    """Handle initial confirmation of both names"""
    
    def name(self) -> Text:
        return "action_confirm_full_name"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        firstname = tracker.get_slot("imprint_firstname")
        surname = tracker.get_slot("imprint_surname")
        
        if not firstname or not surname:
            dispatcher.utter_message(text="I'm missing some information. Let me ask again.")
            return [SlotSet("spelling_stage", None)]
        
        # Create formatted spelling for display
        firstname_spelled = "-".join(list(firstname.upper()))
        surname_spelled = "-".join(list(surname.upper()))
        
        return [
            SlotSet("firstname_spelled", firstname_spelled),
            SlotSet("surname_spelled", surname_spelled),
            SlotSet("spelling_stage", "confirming_both")
        ]

class ActionHandleSpellingConfirmation(Action):
    """Route to next step based on spelling confirmation response"""
    
    def name(self) -> Text:
        return "action_handle_spelling_confirmation"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        intent = tracker.latest_message.get("intent", {}).get("name")
        spelling_stage = tracker.get_slot("spelling_stage")
        
        logger.info(f"Handling confirmation at stage '{spelling_stage}' with intent '{intent}'")
        
        if spelling_stage == "confirming_both":
            if intent == "affirm":
                # Both names confirmed, ask about spelling details
                dispatcher.utter_message(response="utter_great")
                dispatcher.utter_message(response="utter_ask_confirm_spelling")
                return [SlotSet("spelling_stage", "confirming_spelling")]
            else:
                # Names were wrong - jump straight to spelling correction
                dispatcher.utter_message(response="utter_apology_retry")
                dispatcher.utter_message(response="utter_ask_spell_firstname")
                return [SlotSet("spelling_stage", "spelling_first")]
        
        elif spelling_stage == "confirming_spelling":
            if intent == "affirm":
                # Spelling confirmed, we're done!
                #dispatcher.utter_message(response="utter_name_complete")
                dispatcher.utter_message(response="utter_great")
                dispatcher.utter_message(response="utter_pleasure_meet")
                return [
                    SlotSet("spelling_stage", "complete"),
                    SlotSet("name_complete", True)
                ]
            else:
                # Need to correct spelling
                dispatcher.utter_message(response="utter_ask_spell_firstname")
                return [SlotSet("spelling_stage", "spelling_first")]
        
        elif spelling_stage == "confirming_first":
            if intent == "affirm":
                # First name spelling confirmed, move to surname
                dispatcher.utter_message(response="utter_great")
                dispatcher.utter_message(response="utter_ask_spell_surname")
                return [SlotSet("spelling_stage", "spelling_last")]
            else:
                # First name spelling wrong, ask again
                dispatcher.utter_message(response="utter_apology_retry")
                dispatcher.utter_message(response="utter_ask_spell_firstname")
                return [SlotSet("spelling_stage", "spelling_first")]
        
        elif spelling_stage == "confirming_last":
            if intent == "affirm":
                # Last name spelling confirmed, complete!
                #dispatcher.utter_message(response="utter_name_complete")
                dispatcher.utter_message(response="utter_great")
                dispatcher.utter_message(response="utter_pleasure_meet")
                return [
                    SlotSet("spelling_stage", "complete"),
                    SlotSet("name_complete", True)
                ]
            else:
                # Last name spelling wrong, ask again
                dispatcher.utter_message(response="utter_apology_retry")
                dispatcher.utter_message(response="utter_ask_spell_surname")
                return [SlotSet("spelling_stage", "spelling_last")]
        
        return []

class ActionSetImprintName(Action):
    """Combine firstname and surname into imprint_name for database query"""
    
    def name(self) -> Text:
        return "action_set_imprint_name"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        firstname = tracker.get_slot("imprint_firstname")
        surname = tracker.get_slot("imprint_surname")
        
        if firstname and surname:
            imprint_name = f"{firstname}_{surname}"
            logger.info(f"Set imprint_name: {imprint_name}")
            return [SlotSet("imprint_name", imprint_name)]
        else:
            logger.warning(f"Cannot set imprint_name - missing firstname or surname")
            return []
        
class ValidateNameCollectionForm(FormValidationAction):
    """Validate and split names if they contain spaces"""
    
    def name(self) -> Text:
        return "validate_name_collection_form"
    
    def validate_imprint_firstname(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate firstname slot and split if it contains spaces"""
        
        if slot_value and " " in slot_value:
            parts = slot_value.split(None, 1)  # Split on first whitespace only
            new_firstname = parts[0].capitalize()
            new_surname = parts[1].capitalize() if len(parts) > 1 else ""
            
            logger.info(f"Splitting firstname '{slot_value}' into '{new_firstname}' and '{new_surname}'")
            
            # Return both slots if we extracted a surname from firstname
            current_surname = tracker.get_slot("imprint_surname")
            if not current_surname and new_surname:
                return {
                    "imprint_firstname": new_firstname,
                    "imprint_surname": new_surname
                }
            else:
                return {"imprint_firstname": new_firstname}
        
        # If no spaces, just capitalize properly
        if slot_value:
            return {"imprint_firstname": slot_value.capitalize()}
        
        return {"imprint_firstname": slot_value}
    
    def validate_imprint_surname(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate surname slot and capitalize if it contains spaces"""
        
        if slot_value and " " in slot_value:
            # Keep multi-part surnames together (e.g., "Mosier Warren")
            capitalized_surname = " ".join(word.capitalize() for word in slot_value.split())
            logger.info(f"Capitalizing surname '{slot_value}' to '{capitalized_surname}'")
            return {"imprint_surname": capitalized_surname}
        
        # If no spaces, just capitalize properly
        if slot_value:
            return {"imprint_surname": slot_value.capitalize()}
        
        return {"imprint_surname": slot_value}