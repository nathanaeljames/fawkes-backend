from difflib import SequenceMatcher
import re

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_fuzzy_match(spoken_text: str, target_text: str) -> float:
    spoken_normalized = normalize_text(spoken_text)
    target_normalized = normalize_text(target_text)
    matcher = SequenceMatcher(None, spoken_normalized, target_normalized)
    return matcher.ratio()

# Your actual pangram
pangram = "Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."

print("="*80)
print("SequenceMatcher.ratio() BEHAVIOR TEST")
print("="*80)

# Test 1: Perfect repetition
print("\n--- TEST 1: Perfect Repetition ---")
acc1 = "please call stella"
acc2 = "please call stella please call stella"
acc3 = "please call stella please call stella please call stella"
print(f"First time: {calculate_fuzzy_match(acc1, pangram):.1%}")
print(f"2x repeat: {calculate_fuzzy_match(acc2, pangram):.1%}")
print(f"3x repeat: {calculate_fuzzy_match(acc3, pangram):.1%}")

# Test 2: Accumulation with errors
print("\n--- TEST 2: Accumulation with Transcription Errors ---")
good = "please call stella"
with_error = "please call stella stinky farts"
more_error = "please call stella stinky farts dinky hearts"
print(f"Good segment: {calculate_fuzzy_match(good, pangram):.1%}")
print(f"+ garbage: {calculate_fuzzy_match(with_error, pangram):.1%}")
print(f"+ more garbage: {calculate_fuzzy_match(more_error, pangram):.1%}")

# Test 3: Multiple good segments
print("\n--- TEST 3: Multiple Good Segments (Sequential) ---")
seg1 = "please call stella"
seg2 = "please call stella ask her to bring these things"
seg3 = "please call stella ask her to bring these things six spoons"
print(f"Segment 1: {calculate_fuzzy_match(seg1, pangram):.1%}")
print(f"Segment 1+2: {calculate_fuzzy_match(seg2, pangram):.1%}")
print(f"Segment 1+2+3: {calculate_fuzzy_match(seg3, pangram):.1%}")

# Test 4: Your actual log data
print("\n--- TEST 4: Simulating Your Actual Logs ---")
utterances = [
    "please this call stella",
    "ask her to bring these things with her from the store six spoons of fresh snow peas five thick slabs of blue cheese and be a snack for her brother bo",
    "we also need a small plastic snake and a big toy frog for the",
    "she can scoop these things into three red bags we will go meet her wednesday at the train station",
    "please call stella",
    "ask her to bring these things with her from the store",
]

accumulated = ""
for i, utt in enumerate(utterances, 1):
    accumulated += " " + utt
    score = calculate_fuzzy_match(accumulated.strip(), pangram)
    print(f"After utterance {i}: {score:.1%}")
    print(f"  Length ratio: {len(accumulated.strip())}/{len(normalize_text(pangram))}")

# Test 5: THE KEY INSIGHT TEST
print("\n--- TEST 5: Why SequenceMatcher.ratio() Decreases ---")
print("ratio() = 2.0 * matches / (len(text1) + len(text2))")
print()
text1 = "please call stella"
text2 = "please call stella ask her to bring"
text3 = "please call stella ask her to bring stinky farts"
for text in [text1, text2, text3]:
    norm = normalize_text(text)
    pang = normalize_text(pangram)
    matcher = SequenceMatcher(None, norm, pang)
    matches = sum(triple[-1] for triple in matcher.get_matching_blocks())
    total = len(norm) + len(pang)
    ratio = 2.0 * matches / total
    print(f"Text: '{text[:50]}...'")
    print(f"  Matches: {matches}, Total chars: {total}, Ratio: {ratio:.1%}")
    print()