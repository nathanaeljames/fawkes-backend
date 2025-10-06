from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import datetime
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

class ActionTellTime(Action):

    def name(self) -> Text:
        return "action_tell_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        #current_time = datetime.datetime.now().strftime("%H:%M:%S")
        current_time_aware = datetime.datetime.now(EST_TZ)
        current_time = current_time_aware.strftime("%H:%M:%S")
        message = f"Sir, the time is {current_time}"
        
        dispatcher.utter_message(text=message)
        
        return []
    
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