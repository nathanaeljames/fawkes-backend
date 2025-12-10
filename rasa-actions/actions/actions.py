from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
import datetime
from rasa_sdk.types import DomainDict
from rasa_sdk.events import SlotSet, Form, FollowupAction
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
#FASTAPI_HOST = "0.0.0.0"
FASTAPI_HOST = "canary"
FASTAPI_PORT = 9002
# ==========================================================

# SYSTEM TRIGGER PATTERNS
# These are exact-match patterns used to trigger intents programmatically
# and should not be processed as regular user input
SYSTEM_TRIGGERS = [
    "SYSTEM_TRIGGER_ENROLLMENT",
    "SYSTEM_ENROLLMENT_SUCCESS",
    "SYSTEM_ENROLLMENT_ABORT"
]
# ==========================================================

def is_system_trigger(text: str) -> tuple[bool, str]:
    """
    Check if text contains a system trigger pattern.
    Returns: (is_trigger, trigger_name) where trigger_name is the matched pattern or None
    """
    if not text:
        return False, None
    
    text_upper = text.upper()
    for trigger in SYSTEM_TRIGGERS:
        if trigger in text_upper:
            return True, trigger
    
    return False, None

class ActionLogSlots(Action):
    def name(self) -> Text:
        return "action_log_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get current slot values
        ecapa_name = tracker.get_slot("ecapa_name")
        ecapa_firstname = tracker.get_slot("ecapa_firstname")
        ecapa_surname = tracker.get_slot("ecapa_surname")
        ecapa_uid = tracker.get_slot("ecapa_uid")
        imprint_name = tracker.get_slot("imprint_name")
        imprint_firstname = tracker.get_slot("imprint_firstname")
        imprint_surname = tracker.get_slot("imprint_surname")
        imprint_uid = tracker.get_slot("imprint_uid")
        
        # Log to console with timestamp
        logger.info(f"SLOT VALUES - ecapa_name: '{ecapa_name}', ecapa_firstname: '{ecapa_firstname}', ecapa_surname: '{ecapa_surname}', ecapa_uid: '{ecapa_uid}', imprint_name: '{imprint_name}', imprint_firstname: '{imprint_firstname}', imprint_surname: '{imprint_surname}', imprint_uid: '{imprint_uid}'")
        
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
        speaker_name_from_metadata = metadata.get("speaker_name")
        uid_from_metadata = metadata.get("speaker_uid")

        EXCLUDED_SPEAKERS = {"unknown_speaker", "unknown speaker", "unregistered"}
        
        # Only update slots if we have new metadata
        # Otherwise, preserve existing slot values
        if speaker_name_from_metadata is not None and uid_from_metadata is not None:
            ecapa_name = speaker_name_from_metadata
            ecapa_uid = uid_from_metadata
            ecapa_firstname = None
            ecapa_surname = None
            
            # Set ecapa_name and extract firstname/surname if valid
            if ecapa_name and ecapa_name not in EXCLUDED_SPEAKERS:
                parts = ecapa_name.split(" ", 1)  # Split on first SPACE only
                ecapa_firstname = parts[0].capitalize()
                if len(parts) > 1:
                    ecapa_surname = parts[1].capitalize()
            
            logger.info(f"ActionSetNameSlots - Full metadata: {metadata}")
            logger.info(f"ActionSetNameSlots - ecapa_name from metadata: '{ecapa_name}'")
            if ecapa_firstname:
                logger.info(f"ActionSetNameSlots - Extracted firstname: '{ecapa_firstname}', surname: '{ecapa_surname}'")
            else:
                logger.info(f"ActionSetNameSlots - ecapa_name excluded or None: '{ecapa_name}'")
            
            return [
                SlotSet("ecapa_name", ecapa_name),
                SlotSet("ecapa_firstname", ecapa_firstname),
                SlotSet("ecapa_surname", ecapa_surname),
                SlotSet("ecapa_uid", ecapa_uid)
            ]
        else:
            # No metadata - preserve existing slot values by returning empty list
            logger.info(f"ActionSetNameSlots - No metadata, preserving existing ecapa slots")
            return []

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
class ActionHandleNameInput(Action):
    """Router action that handles name input and decides whether to trigger enrollment"""
    
    def name(self) -> Text:
        return "action_handle_name_input"
    
    def _check_name_solicited(self, tracker: Tracker) -> bool:
        """Check if the previous bot utterance asked for the user's name"""
        NAME_SOLICITING_UTTERANCES = [
            "utter_tell_name",
            "utter_ask_familar",
            "utter_ask_full_name"
        ]
        
        # Look backwards through events to find the last bot utterance
        for event in reversed(tracker.events):
            if event.get("event") == "bot":
                last_bot_utterance = event.get("metadata", {}).get("utter_action")
                if last_bot_utterance in NAME_SOLICITING_UTTERANCES:
                    logger.info(f"Name was solicited - previous utterance: {last_bot_utterance}")
                    return True
                # If we found a bot utterance but it wasn't name-soliciting, stop looking
                break
        
        logger.info("Name was not solicited")
        return False
    
    def _check_exact_pattern(self, text: str) -> bool:
        """Check if text matches exact 'my name is X' or 'my name is X Y' pattern"""
        import re
        
        # Pattern matches "my name is [word]" or "my name is [word] [word]"
        pattern = r"^my name is \w+(?: \w+)+$"
        match = re.match(pattern, text.lower().strip())
        
        if match:
            logger.info(f"Exact name pattern matched: '{text}'")
            return True
        
        logger.info(f"Exact name pattern did not match: '{text}'")
        return False
    
    def _check_ecapa_match(self, text: str, ecapa_firstname: str, ecapa_surname: str, ecapa_name: str) -> bool:
        """Check if user text contains a match or close match to ecapa name components"""
        from difflib import SequenceMatcher
        
        if not ecapa_firstname and not ecapa_name:
            logger.info("No ecapa name data available for comparison")
            return False
        
        text_lower = text.lower()
        
        # Check for exact substring matches first
        if ecapa_firstname and ecapa_firstname.lower() in text_lower:
            if ecapa_surname and ecapa_surname.lower() in text_lower:
                logger.info(f"Exact match found: both '{ecapa_firstname}' and '{ecapa_surname}' in text")
                return True
        
        # Check for fuzzy matches (allowing 1-2 character differences)
        def fuzzy_match(word1: str, word2: str, threshold: float = 0.8) -> bool:
            """Return True if words are similar enough (accounts for 1-2 letter differences)"""
            if not word1 or not word2:
                return False
            ratio = SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
            return ratio >= threshold
        
        # Extract words from user text (simple tokenization)
        words = text_lower.split()
        
        # Check if both firstname and surname have fuzzy matches in the text
        firstname_matched = False
        surname_matched = False
        
        if ecapa_firstname:
            for word in words:
                if fuzzy_match(word, ecapa_firstname):
                    firstname_matched = True
                    break
        
        if ecapa_surname:
            for word in words:
                if fuzzy_match(word, ecapa_surname):
                    surname_matched = True
                    break
        
        if firstname_matched and surname_matched:
            logger.info(f"Fuzzy match found: both '{ecapa_firstname}' and '{ecapa_surname}' matched in text")
            return True
        
        logger.info(f"No ecapa name match found in text")
        return False
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get current context
        user_text = tracker.latest_message.get("text", "")
        intent = tracker.latest_message.get("intent", {}).get("name")
        
        ecapa_firstname = tracker.get_slot("ecapa_firstname")
        ecapa_surname = tracker.get_slot("ecapa_surname")
        ecapa_name = tracker.get_slot("ecapa_name")
        ecapa_uid = tracker.get_slot("ecapa_uid")
        enrollment_active = tracker.get_slot("enrollment_active")
        
        logger.info(f"Routing name input - intent: {intent}, text: '{user_text}'")
        logger.info(f"Context - ecapa_name: {ecapa_name}, enrollment_active: {enrollment_active}")
        
        # Evaluate conditions
        name_solicited = self._check_name_solicited(tracker)
        exact_pattern = self._check_exact_pattern(user_text)
        ecapa_match = self._check_ecapa_match(user_text, ecapa_firstname, ecapa_surname, ecapa_name)
        
        logger.info(f"Conditions - solicited: {name_solicited}, exact_pattern: {exact_pattern}, ecapa_match: {ecapa_match}")
        
        # Routing logic
        if ecapa_match:
            logger.info("Route: Name matches ecapa_name - greeting existing user")
            dispatcher.utter_message(response="utter_pleasure_meet_again")
            #return []
            #return [SlotSet("imprint_firstname", ecapa_firstname), SlotSet("imprint_surname", ecapa_surname), SlotSet("imprint_uid", ecapa_uid)]
            # No need to query the database at this point, reset enrollment active flag (if it is set)
            return [
                SlotSet("imprint_firstname", ecapa_firstname), 
                SlotSet("imprint_surname", ecapa_surname), 
                SlotSet("imprint_uid", ecapa_uid),
                FollowupAction("action_reset_enrollment")
            ]
        
        elif name_solicited:
            logger.info("Route: Name was solicited - beginning enrollment")
            # This will trigger the name_collection_form rule
            #return [SlotSet("imprint_name_provided", True)]
            return [Form("name_collection_form")]
        
        #elif enrollment_active and exact_pattern:
        #    logger.info("Route: Enrollment active with exact pattern - beginning enrollment")
        #    return [SlotSet("imprint_name_provided", True)]

        elif exact_pattern:
            logger.info("Route: Exact name pattern used with no matching ecapa - beginning enrollment")
            #return [SlotSet("imprint_name_provided", True)]
            return [Form("name_collection_form")]
        
        else:
            logger.info("Route: No enrollment triggered - user likely mentioned someone else's name")
            return []

class ActionProcessSpelling(Action):
    """Process spelling input in various formats and convert to word"""
    
    def name(self) -> Text:
        return "action_process_spelling"
    
    def _normalize_spelling_input(self, text: str) -> tuple:
        """
        Normalize spelling input from various formats.
        Returns: (processed_name, formatted_spelling, is_valid)
        """
        # Convert to lowercase for easier matching
        text_lower = text.lower().strip()
        
        logger.info(f"Starting normalization of: '{text_lower}'")
        
        # First, tokenize the input into parts (split by spaces and hyphens)
        # We need to preserve the structure to handle "space", "apostrophe", etc.
        tokens = []
        current_token = ""
        
        for char in text_lower:
            if char in [' ', '-']:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        logger.info(f"Tokenized into: {tokens}")
        
        # Now process tokens - convert special words to their characters
        special_words = {
            'space': ' ',
            'apostrophe': "'",
            'dash': '-',
            'hyphen': '-'
        }
        
        processed_chars = []
        for token in tokens:
            if token in special_words:
                # This is a special character word
                processed_chars.append(special_words[token])
                logger.info(f"Converted '{token}' to '{special_words[token]}'")
            elif len(token) == 1 and token.isalpha():
                # Single letter
                processed_chars.append(token.upper())
            elif token.isalpha() and len(token) <= 3:
                # Short sequence of letters - might be letters run together
                # Treat each as a separate letter
                for char in token:
                    processed_chars.append(char.upper())
                logger.info(f"Split multi-letter token '{token}' into individual letters")
            else:
                # Invalid token
                logger.warning(f"Unrecognized token: '{token}'")
                return None, None, False
        
        logger.info(f"Processed characters: {processed_chars}")
        
        if not processed_chars:
            return None, None, False
        
        # Build the actual name with proper capitalization
        name_parts = []
        current_part = []
        
        for char in processed_chars:
            if char == ' ':
                # Space indicates a new word/part
                if current_part:
                    name_parts.append(''.join(current_part))
                    current_part = []
            elif char in ["'", '-']:
                # Apostrophe or hyphen stays in the current part
                current_part.append(char)
            else:
                # Letter
                current_part.append(char)
        
        # Don't forget the last part
        if current_part:
            name_parts.append(''.join(current_part))
        
        logger.info(f"Name parts before capitalization: {name_parts}")
        
        # Capitalize each part properly
        capitalized_parts = []
        for part in name_parts:
            # Capitalize the first letter and after apostrophes/hyphens
            result = []
            capitalize_next = True
            for char in part:
                if char.isalpha():
                    if capitalize_next:
                        result.append(char.upper())
                        capitalize_next = False
                    else:
                        result.append(char.lower())
                else:
                    result.append(char)
                    if char in ["'", '-']:
                        capitalize_next = True
            capitalized_parts.append(''.join(result))
        
        # Join parts with spaces
        processed_name = ' '.join(capitalized_parts)
        
        logger.info(f"Final processed name: '{processed_name}'")
        
        # Create formatted spelling for display
        formatted_parts = []
        for char in processed_chars:
            if char == ' ':
                formatted_parts.append('space')
            elif char == "'":
                formatted_parts.append('apostrophe')
            elif char == '-':
                formatted_parts.append('hyphen')
            else:
                formatted_parts.append(char.upper())
        
        formatted_spelling = '-'.join(formatted_parts)
        
        logger.info(f"Formatted spelling: '{formatted_spelling}'")
        
        # Validate: must have at least one letter
        has_letter = any(c.isalpha() for c in processed_name)
        is_reasonable_length = 2 <= len(processed_name) <= 50
        
        return processed_name, formatted_spelling, has_letter and is_reasonable_length
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        text = tracker.latest_message.get("text", "").strip()
        intent = tracker.latest_message.get("intent", {}).get("name")
        spelling_stage = tracker.get_slot("spelling_stage")

        # Reject system trigger patterns
        is_trigger, trigger_name = is_system_trigger(text)
        if is_trigger:
            logger.warning(f"Ignoring system trigger '{trigger_name}' in spelling action")
            return []
        
        logger.info(f"Processing spelling input: '{text}' (intent: {intent}) at stage: {spelling_stage}")
        
        # Validate that we got the spell_name intent
        if intent != "spell_name":
            logger.warning(f"Expected spell_name intent but got {intent}")
            dispatcher.utter_message(text="Please spell your name letter-by-letter, like N-A-T-E.")
            return []
        
        # Process the spelling input
        processed_name, formatted_spelling, is_valid = self._normalize_spelling_input(text)
        
        if not is_valid:
            logger.warning(f"Invalid spelling format: '{text}'")
            dispatcher.utter_message(text="I didn't understand that spelling. Please spell it letter-by-letter.")
            return []
        
        logger.info(f"Converted spelling '{text}' to name '{processed_name}' (formatted: {formatted_spelling})")
        
        if spelling_stage == "spelling_first":
            dispatcher.utter_message(text=f"{processed_name}, spelled {formatted_spelling}. Did I get that right?")
            return [
                SlotSet("imprint_firstname", processed_name),
                SlotSet("imprint_firstname_spelled", formatted_spelling),
                SlotSet("spelling_stage", "confirming_first")
            ]
        elif spelling_stage == "spelling_last":
            dispatcher.utter_message(text=f"{processed_name}, spelled {formatted_spelling}. Did I get that right?")
            return [
                SlotSet("imprint_surname", processed_name),
                SlotSet("imprint_surname_spelled", formatted_spelling),
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
        imprint_firstname_spelled = "-".join(list(firstname.upper()))
        imprint_surname_spelled = "-".join(list(surname.upper()))
        
        return [
            SlotSet("imprint_firstname_spelled", imprint_firstname_spelled),
            SlotSet("imprint_surname_spelled", imprint_surname_spelled),
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
                #dispatcher.utter_message(response="utter_pleasure_meet")
                return [
                    SlotSet("spelling_stage", "complete"),
                    SlotSet("name_complete", True),
                    FollowupAction("action_query_userbase") 
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
                #dispatcher.utter_message(response="utter_pleasure_meet")
                return [
                    SlotSet("spelling_stage", "complete"),
                    SlotSet("name_complete", True),
                    FollowupAction("action_query_userbase") 
                ]
            else:
                # Last name spelling wrong, ask again
                dispatcher.utter_message(response="utter_apology_retry")
                dispatcher.utter_message(response="utter_ask_spell_surname")
                return [SlotSet("spelling_stage", "spelling_last")]
        
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

        # Reject system trigger patterns
        if slot_value:
            is_trigger, trigger_name = is_system_trigger(slot_value)
            if is_trigger:
                logger.warning(f"Ignoring system trigger '{trigger_name}' in firstname validation")
                return {"imprint_firstname": None}
        
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

        # Reject system trigger patterns
        if slot_value:
            is_trigger, trigger_name = is_system_trigger(slot_value)
            if is_trigger:
                logger.warning(f"Ignoring system trigger '{trigger_name}' in surname validation")
                return {"imprint_surname": None}
        
        if slot_value and " " in slot_value:
            # Keep multi-part surnames together (e.g., "Mosier Warren")
            capitalized_surname = " ".join(word.capitalize() for word in slot_value.split())
            logger.info(f"Capitalizing surname '{slot_value}' to '{capitalized_surname}'")
            return {"imprint_surname": capitalized_surname}
        
        # If no spaces, just capitalize properly
        if slot_value:
            return {"imprint_surname": slot_value.capitalize()}
        
        return {"imprint_surname": slot_value}

class ActionQueryUserbase(Action):
    def name(self) -> Text:
        return "action_query_userbase"
    
    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        voiceclone_active = tracker.get_slot("voiceclone_active")
        fastapi_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/query"

        if voiceclone_active:
            # Voice cloning flow - use voiceclone_lazyname for fuzzy matching
            lazyname = tracker.get_slot("voiceclone_lazyname")
            
            if not lazyname:
                dispatcher.utter_message(response="utter_ask_whose_voice")
                return [SlotSet("voiceclone_lazyname", None)]
            
            logger.info(f"Querying database for: {lazyname} (context: voiceclone, fuzzy match)")
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        fastapi_url,
                        json={
                            "full_name": lazyname
                        },
                        timeout=5
                    ) as response:
                        
                        if response.status == HTTPStatus.OK:
                            data = await response.json()
                            
                            if data.get('status') == 'not_found':
                                # No speakers found
                                dispatcher.utter_message(response="utter_voiceclone_abort")
                                return [
                                    SlotSet("voiceclone_lazyname", None),
                                    FollowupAction("action_reset_voice_cloning")
                                ]
                            
                            confidence = data.get('confidence', 0)
                            speaker_name = data.get('speaker_name')  # Backend format: firstname_surname
                            firstname = data.get('firstname')
                            surname = data.get('surname')
                            matched_name = f"{firstname} {surname}"  # Human readable
                            
                            logger.info(f"Speaker match: {matched_name} (confidence: {confidence})")
                            
                            # ALWAYS set candidate and verified slots first
                            events = [
                                SlotSet("voiceclone_candidate", matched_name),
                                SlotSet("voiceclone_speakername", speaker_name),
                                SlotSet("voiceclone_confidence", confidence),
                                SlotSet("voiceclone_lazyname", None)
                            ]
                            
                            if confidence >= 0.95:
                                # High confidence: proceed automatically
                                #dispatcher.utter_message(response="utter_voiceclone_proceed")
                                dispatcher.utter_message(text=f"Proceeding to voice clone with {matched_name}")
                                #events.append(SlotSet("voiceclone_verified", speaker_name))
                                events.append(SlotSet("passagecollect_active", True))
                                events.append(SlotSet("voiceclone_retry_count", 0))
                                return events
                            
                            elif confidence >= 0.70:
                                # Medium confidence: ask for verification
                                #dispatcher.utter_message(response="utter_confirm_speaker_match")
                                dispatcher.utter_message(text=f"Did you mean {matched_name}?")
                                return events
                            
                            else:
                                # Low confidence: retry once then abort
                                retry_count = tracker.get_slot("voiceclone_retry_count") or 0
                                
                                if retry_count < 2:  # Changed from 1 to 2
                                    dispatcher.utter_message(response="utter_ask_voiceclone_retry")
                                    return [
                                        SlotSet("voiceclone_lazyname", None),
                                        SlotSet("voiceclone_retry_count", retry_count + 1)
                                    ]
                                else:
                                    # Max retries exceeded - abort
                                    dispatcher.utter_message(response="utter_voiceclone_abort")
                                    return [
                                        SlotSet("voiceclone_lazyname", None),
                                        SlotSet("voiceclone_retry_count", 0),
                                        FollowupAction("action_reset_voice_cloning")
                                    ]
                        
                        else:
                            logger.error(f"Server returned status {response.status}")
                            dispatcher.utter_message(text="I'm having trouble looking up that speaker. Please try again.")
                            return [SlotSet("voiceclone_lazyname", None)]
            
            except Exception as e:
                logger.error(f"Error querying userbase: {e}")
                dispatcher.utter_message(text="I encountered an error looking up that speaker. Please try again.")
                return [SlotSet("voiceclone_lazyname", None)]
        
        else:
            # Enrollment flow - unchanged
            query_firstname = tracker.get_slot("imprint_firstname")
            query_surname = tracker.get_slot("imprint_surname")
            
            if not query_firstname or not query_surname:
                logger.warning(f"Missing name information - firstname: {query_firstname}, surname: {query_surname}")
                return [
                    SlotSet("imprint_uid", None),
                    FollowupAction("action_handle_enrollment_routing")
                ]
            
            logger.info(f"Querying database for: {query_firstname} {query_surname} (context: enrollment, exact match)")
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        fastapi_url,
                        json={
                            "firstname": query_firstname,
                            "surname": query_surname
                        },
                        timeout=5
                    ) as response:
                        
                        if response.status == HTTPStatus.OK:
                            result = await response.json()
                            uid = result.get("uid")
                            firstname = result.get("firstname")
                            surname = result.get("surname")
                            
                            if uid and firstname and surname:
                                logger.info(f"Found speaker {firstname} {surname} with UID: {uid}")
                                imprint_name = f"{firstname} {surname}"
                                return [
                                    SlotSet("imprint_uid", str(uid)),
                                    SlotSet("imprint_name", imprint_name),
                                    FollowupAction("action_handle_enrollment_routing")
                                ]
                            else:
                                logger.info(f"No record found for {query_firstname} {query_surname}")
                                return [
                                    SlotSet("imprint_uid", None),
                                    FollowupAction("action_handle_enrollment_routing")
                                ]
                        
                        elif response.status == HTTPStatus.NOT_FOUND:
                            logger.info(f"No record found for {query_firstname} {query_surname} (404)")
                            return [
                                SlotSet("imprint_uid", None),
                                FollowupAction("action_handle_enrollment_routing")
                            ]
                        
                        else:
                            logger.error(f"Server returned status {response.status}")
                            return [
                                SlotSet("imprint_uid", None),
                                FollowupAction("action_handle_enrollment_routing")
                            ]
            
            except Exception as e:
                logger.error(f"Error querying userbase: {e}")
                return [
                    SlotSet("imprint_uid", None),
                    FollowupAction("action_handle_enrollment_routing")
                ]

class ActionTriggerEnrollment(Action):
    def name(self) -> Text:
        return "action_trigger_enrollment"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Whatever else this action does...
        logger.info("Enrollment triggered")
        
        # Add the slot set
        return [SlotSet("enrollment_active", True)]

class ActionHandleEnrollmentRouting(Action):
    """Handle routing after userbase query based on whether user was found"""
    
    def name(self) -> Text:
        return "action_handle_enrollment_routing"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        imprint_uid = tracker.get_slot("imprint_uid")
        
        if imprint_uid:
            # User found in database - existing user path
            logger.info(f"User found in database with UID: {imprint_uid}")
            dispatcher.utter_message(response="utter_pleasure_meet_again")
            dispatcher.utter_message(response="utter_weak_imprint")
            dispatcher.utter_message(response="utter_ask_recite_prompt")
        else:
            # User not found - new user path
            logger.info("User not found in database - new user")
            dispatcher.utter_message(response="utter_pleasure_meet")
            dispatcher.utter_message(response="utter_no_record")
            dispatcher.utter_message(response="utter_ask_recite_prompt")
        
        return []

class ActionStartEnrollmentRecording(Action):
    def name(self) -> Text:
        return "action_start_enrollment_recording"
    
    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        sender_id = tracker.sender_id
        imprint_uid = tracker.get_slot("imprint_uid")
        imprint_firstname = tracker.get_slot("imprint_firstname")
        imprint_surname = tracker.get_slot("imprint_surname")
        
        # Trigger recording on server
        fastapi_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/record_pangram"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    fastapi_url,
                    json={
                        "sender_id": sender_id,
                        "imprint_uid": imprint_uid,
                        "imprint_firstname": imprint_firstname,
                        "imprint_surname": imprint_surname
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

        # Determine status based on the triggering intent
        latest_intent = tracker.latest_message.get('intent', {}).get('name', '')
        
        if latest_intent == 'system_enrollment_complete_success':
            status = 'success'
        else:
            status = 'aborted'  # User cancelled or other reason

        logger.info(f"Attempting to reset enrollment for client {sender_id} via FastAPI endpoint: {fastapi_url} with status: {status}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    fastapi_url, 
                    json={"client_id": sender_id, "status": status},
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
        return [
            SlotSet("enrollment_active", False)
            #FollowupAction("action_listen")  # Explicitly tell Rasa to just listen
        ]

# Voice clone actions
class ActionStartVoiceCloning(Action):
    """Initiates voice cloning workflow"""
    
    def name(self) -> Text:
        return "action_start_voice_cloning"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        return [SlotSet("voiceclone_active", True)]

class ActionSetVoicecloneLazyname(Action):
    """Capture user utterance for voice cloning when voiceclone_active=true"""
    def name(self) -> Text:
        return "action_set_voiceclone_lazyname"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        voiceclone_active = tracker.get_slot("voiceclone_active")
        
        if not voiceclone_active:
            # Not in voice cloning flow, do nothing
            return []
        
        # Get the user's last message text
        user_text = tracker.latest_message.get('text', '').strip()
        
        if not user_text:
            logger.warning("No text in user message for voiceclone_lazyname")
            return []
        
        logger.info(f"Captured voiceclone_lazyname: {user_text}")
        return [SlotSet("voiceclone_lazyname", user_text)]

class ActionConfirmSpeakerMatch(Action):
    """User confirmed the speaker match"""
    def name(self) -> Text:
        return "action_confirm_speaker_match"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Candidate and verified already set by ActionQueryUserbase
        # Just proceed with confirmation message
        dispatcher.utter_message(response="utter_voiceclone_proceed")
        
        return [
            SlotSet("voiceclone_confidence", None),
            SlotSet("passagecollect_active", True),
            SlotSet("voiceclone_retry_count", 0)
        ]

class ActionRejectSpeakerMatch(Action):
    """User rejected the speaker match - retry once then abort"""
    def name(self) -> Text:
        return "action_reject_speaker_match"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        retry_count = tracker.get_slot("voiceclone_retry_count") or 0
        
        # Clear candidate and verified
        events = [
            SlotSet("voiceclone_candidate", None),
            SlotSet("voiceclone_speakername", None),
            SlotSet("voiceclone_confidence", None)
        ]
        
        if retry_count < 2:
            dispatcher.utter_message(response="utter_ask_voiceclone_retry")
            events.append(SlotSet("voiceclone_retry_count", retry_count + 1))
        else:
            dispatcher.utter_message(response="utter_voiceclone_abort")
            events.extend([
                SlotSet("voiceclone_retry_count", 0),
                FollowupAction("action_reset_voice_cloning")
            ])
        
        return events

class ActionResetVoiceCloning(Action):
    """Reset voice cloning slots and state"""
    
    def name(self) -> Text:
        return "action_reset_voice_cloning"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info("Resetting voice cloning workflow")
        
        return [
            # Voice cloning speaker selection slots
            SlotSet("voiceclone_active", False),
            SlotSet("voiceclone_lazyname", None),
            SlotSet("voiceclone_candidate", None),
            SlotSet("voiceclone_speakername", None),
            SlotSet("voiceclone_confidence", None),
            SlotSet("voiceclone_retry_count", 0),
            # Passage selection slots
            SlotSet("passagecollect_active", False),
            SlotSet("psource_lazystring", None),
            SlotSet("psource_candidate", None),
            SlotSet("psource_verified", None),
            SlotSet("psource_confidence", None),
            SlotSet("psource_retry_count", 0),
            SlotSet("available_passage_sources", []),
            SlotSet("selected_quote", None),
            # Return to listening
            FollowupAction("action_listen")
        ]

class ActionQueryPassages(Action):
    """
    Unified action to query passages table.
    Handles: unique_sources, match_source, select_quote
    Mirrors action_query_userbase pattern.
    """
    
    def name(self) -> Text:
        return "action_query_passages"
    
    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Determine which action to take based on workflow state
        psource_lazystring = tracker.get_slot("psource_lazystring")
        psource_verified = tracker.get_slot("psource_verified")
        available_sources = tracker.get_slot("available_passage_sources")
        
        fastapi_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/passages/query"
        
        # Step 1: Get unique sources (no lazystring, no verified, no available sources)
        if not psource_lazystring and not psource_verified and not available_sources:
            action_type = "unique_sources"
            payload = {"action": action_type}
            
        # Step 2: Match source (has lazystring, no verified)
        elif psource_lazystring and not psource_verified:
            action_type = "match_source"
            payload = {
                "action": action_type,
                "fuzzy_source": psource_lazystring
            }
            
        # Step 3: Select quote (has verified)
        elif psource_verified:
            action_type = "select_quote"
            payload = {
                "action": action_type,
                "source_name": psource_verified
            }
        else:
            logger.error("ActionQueryPassages: Invalid state")
            return []
        
        logger.info(f"ActionQueryPassages: {action_type} - payload: {payload}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    fastapi_url,
                    json=payload,
                    timeout=5
                ) as response:
                    
                    if response.status == HTTPStatus.OK:
                        result = await response.json()
                        
                        # Handle unique_sources response
                        if action_type == "unique_sources":
                            sources = result.get("sources", [])
                            
                            if len(sources) == 0:
                                dispatcher.utter_message(text="I don't have any passages available.")
                                return []
                            
                            # Format for natural speech
                            if len(sources) == 1:
                                formatted = sources[0]
                            elif len(sources) == 2:
                                formatted = f"{sources[0]} or {sources[1]}"
                            else:
                                formatted = ", ".join(sources[:-1]) + f", or {sources[-1]}"
                            
                            logger.info(f"Found {len(sources)} passage sources")
                            return [SlotSet("available_passage_sources", formatted)]
                        
                        # Handle match_source response
                        elif action_type == "match_source":
                            source_name = result.get("source_name")
                            confidence = result.get("confidence", 0.0)
                            
                            if source_name:
                                logger.info(f"Matched '{psource_lazystring}' → '{source_name}' ({confidence:.2%})")
                                return [
                                    SlotSet("psource_candidate", source_name),
                                    SlotSet("psource_confidence", confidence)
                                ]
                            else:
                                logger.warning(f"No match found for '{psource_lazystring}'")
                                return [
                                    SlotSet("psource_candidate", None),
                                    SlotSet("psource_confidence", 0.0)
                                ]
                        
                        # Handle select_quote response
                        elif action_type == "select_quote":
                            quote = result.get("quote")
                            
                            if quote:
                                logger.info(f"Selected quote from '{psource_verified}': {quote[:60]}...")
                                return [SlotSet("selected_quote", quote)]
                            else:
                                logger.error(f"No quote found for '{psource_verified}'")
                                dispatcher.utter_message(text="I couldn't find a quote from that source.")
                                return []
                    
                    else:
                        logger.error(f"Passages query failed: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error querying passages: {e}")
            return []

class ActionSetPsourceLazystring(Action):
    """Capture ANY text as passage source name"""
    
    def name(self) -> Text:
        return "action_set_psource_lazystring"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        passagecollect_active = tracker.get_slot("passagecollect_active")
        
        if not passagecollect_active:
            # Not in passage collection flow, do nothing
            return []

        user_text = tracker.latest_message.get('text', '').strip()
        if not user_text:
            logger.warning("No text in user message for psource_lazystring")
            return []
        
        logger.info(f"Captured psource_lazystring: {user_text}")
        return [SlotSet("psource_lazystring", user_text)]

class ActionHandlePassageMatch(Action):
    """
    Handle the fuzzy match result for passage source.
    Routes based on confidence: >=95% auto-verify, >=70% confirm, <70% retry
    Mirrors action_handle_speaker_match pattern.
    """
    
    def name(self) -> Text:
        return "action_handle_passage_match"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        candidate = tracker.get_slot("psource_candidate")
        confidence = tracker.get_slot("psource_confidence") or 0.0
        retry_count = tracker.get_slot("psource_retry_count") or 0
        lazystring = tracker.get_slot("psource_lazystring")
        
        logger.info(f"Passage match - candidate: {candidate}, confidence: {confidence:.2%}, retry: {retry_count}")
        
        # No candidate found
        if not candidate:
            if retry_count >= 2:
                # Abort after 2 retries
                logger.info("Passage match abort - max retries reached")
                dispatcher.utter_message(response="utter_psource_abort")
                return [
                    FollowupAction("action_reset_passage_selection")
                ]
            else:
                # Retry
                logger.info("Passage match retry")
                dispatcher.utter_message(response="utter_ask_passage_retry")
                dispatcher.utter_message(response="utter_ask_passage_source")
                return [
                    SlotSet("psource_candidate", None),
                    SlotSet("psource_confidence", None),
                    SlotSet("psource_lazystring", None),
                    SlotSet("psource_retry_count", retry_count + 1)
                ]
        
        # High confidence - auto verify
        if confidence >= 0.95:
            logger.info(f"High confidence match ({confidence:.2%}) - auto-verifying '{candidate}'")
            return [
                SlotSet("psource_verified", candidate),
                SlotSet("psource_retry_count", 0),
                SlotSet("psource_lazystring", None),
                # Trigger quote selection
                FollowupAction("action_query_passages")
            ]
        
        # Medium confidence - ask for confirmation
        elif confidence >= 0.70:
            logger.info(f"Medium confidence match ({confidence:.2%}) - confirming '{candidate}'")
            dispatcher.utter_message(response="utter_confirm_passage_source")
            return []
        
        # Low confidence - retry
        else:
            if retry_count >= 2:
                # Abort after 2 retries
                logger.info(f"Low confidence ({confidence:.2%}) and max retries - aborting")
                dispatcher.utter_message(response="utter_psource_abort")
                return [
                    FollowupAction("action_reset_passage_selection")
                ]
            else:
                # Retry
                logger.info(f"Low confidence ({confidence:.2%}) - retry {retry_count + 1}")
                dispatcher.utter_message(response="utter_ask_passage_retry")
                dispatcher.utter_message(response="utter_ask_passage_source")
                return [
                    SlotSet("psource_candidate", None),
                    SlotSet("psource_confidence", None),
                    SlotSet("psource_lazystring", None),
                    SlotSet("psource_retry_count", retry_count + 1)
                ]

class ActionConfirmPassageMatch(Action):
    """User confirmed the passage source match"""
    
    def name(self) -> Text:
        return "action_confirm_passage_match"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        candidate = tracker.get_slot("psource_candidate")
        
        logger.info(f"User confirmed passage source: '{candidate}'")
        
        return [
            SlotSet("psource_verified", candidate),
            SlotSet("psource_candidate", None),
            SlotSet("psource_confidence", None),
            SlotSet("psource_lazystring", None),
            SlotSet("psource_retry_count", 0),
            # Trigger quote selection
            FollowupAction("action_query_passages")
        ]

class ActionRejectPassageMatch(Action):
    """User rejected the passage source match - retry"""
    
    def name(self) -> Text:
        return "action_reject_passage_match"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        retry_count = tracker.get_slot("psource_retry_count") or 0
        
        # Clear candidate
        events = [
            SlotSet("psource_candidate", None),
            SlotSet("psource_confidence", None),
            SlotSet("psource_lazystring", None)
        ]
        
        if retry_count < 2:
            logger.info(f"User rejected match - retry {retry_count + 1}")
            dispatcher.utter_message(response="utter_ask_passage_retry")
            dispatcher.utter_message(response="utter_ask_passage_source")
            events.append(SlotSet("psource_retry_count", retry_count + 1))
        else:
            logger.info("User rejected match - max retries, aborting")
            dispatcher.utter_message(response="utter_psource_abort")
            events.extend([
                SlotSet("psource_retry_count", 0),
                FollowupAction("action_reset_passage_selection")
            ])
        
        return events

class ActionResetPassageSelection(Action):
    """Reset all passage selection slots"""
    
    def name(self) -> Text:
        return "action_reset_passage_selection"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info("Resetting passage selection workflow")
        
        return [
            SlotSet("psource_lazystring", None),
            SlotSet("psource_candidate", None),
            SlotSet("psource_verified", None),
            SlotSet("psource_confidence", None),
            SlotSet("psource_retry_count", 0),
            SlotSet("selected_quote", None),
            # Don't reset available_passage_sources - may want to reuse
        ]