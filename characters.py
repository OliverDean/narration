import os
import json
import spacy
import re

all_speakers = [
    "p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", "p233", "p234",
    "p236", "p237", "p238", "p239", "p240", "p241", "p243", "p244", "p245", "p246",
    "p247", "p248", "p249", "p250", "p251", "p252", "p253", "p254", "p255", "p256",
    "p257", "p258", "p259", "p260", "p261", "p262", "p263", "p264", "p265", "p266",
    "p267", "p268", "p269", "p270", "p271", "p272", "p273", "p274", "p275", "p276",
    "p277", "p278", "p279", "p280", "p281", "p282", "p283", "p284", "p285", "p286",
    "p287", "p288", "p292", "p293", "p294", "p295", "p297", "p298", "p299", "p300",
    "p301", "p302", "p303", "p304", "p305", "p306", "p307", "p308", "p310", "p311",
    "p312", "p313", "p314", "p316", "p317", "p318", "p323", "p326", "p329", "p330",
    "p333", "p334", "p335", "p336", "p339", "p340", "p341", "p343", "p345", "p347",
    "p351", "p360", "p361", "p362", "p363", "p364", "p374", "p376"
]

def unify_character_name(raw_name: str, synonyms_map: dict) -> str:
    """
    Example unify function:
      - Title-cases name,
      - if matches synonyms or partial, unify to canonical.
    """
    candidate = raw_name.strip().title()
    if candidate in synonyms_map:
        return candidate
    for main_name, variants in synonyms_map.items():
        if candidate == main_name.title():
            return main_name
        for v in variants:
            if v.lower() in candidate.lower():
                return main_name
    return candidate

def identify_characters_in_text(text_file: str,
                                synonyms_json="character_synonyms.json",
                                voice_map_json="character_voices.json"):
    """
    1) Reads text_file.
    2) Loads synonyms + voice map.
    3) spaCy NLP on text -> find PERSON ents.
    4) unify_character_name(...) for each entity.
    5) If not in voice_map, assign next round-robin speaker ID from all_speakers.
    6) Save voice_map if new chars were added.
    """

    if not os.path.isfile(text_file):
        print(f"[ERROR] File not found: {text_file}")
        return

    with open(text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Load synonyms
    synonyms_map = {}
    if os.path.isfile(synonyms_json):
        with open(synonyms_json, "r", encoding="utf-8") as f:
            synonyms_map = json.load(f)

    # Load or create voice map
    voice_map = {}
    if os.path.isfile(voice_map_json):
        with open(voice_map_json, "r", encoding="utf-8") as f:
            voice_map = json.load(f)

    # Initialize spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(raw_text)

    # We track an index to do round-robin through all_speakers
    # Also track what IDs are already used.
    used_ids = set()
    for v in voice_map.values():
        if "speaker_id" in v:
            used_ids.add(v["speaker_id"])

    # We'll store the next speaker index in voice_map metadata if we want
    # or you could store it globally. For a quick approach, we just start at 0
    # each run, or do something else. We'll do a local pointer: next_index=0
    next_index = 0

    def get_next_round_robin_id():
        nonlocal next_index
        attempts = 0
        while attempts < len(all_speakers):
            candidate = all_speakers[next_index % len(all_speakers)]
            next_index += 1
            attempts += 1
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate
        # If we cycle all_speakers and everything is used, we do not add new IDs
        # or we pick one forcibly. Let's forcibly pick the last candidate.
        return all_speakers[-1]

    new_chars = False

    # For each PERSON entity
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            unified = unify_character_name(ent.text, synonyms_map)
            if len(unified) < 2:
                continue  # skip trivial or single letters

            if unified not in voice_map:
                # assign next round-robin ID
                new_id = get_next_round_robin_id()
                voice_map[unified] = {
                    "speaker_id": new_id,
                    "description": f"Round-robin assigned voice for {unified}"
                }
                new_chars = True

    if new_chars:
        with open(voice_map_json, "w", encoding="utf-8") as f:
            json.dump(voice_map, f, indent=2)
        print(f"[INFO] Updated {voice_map_json} with newly discovered characters.")
    else:
        print("[INFO] No new characters discovered or all were already in voice map.")


# Optional simple CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Identify new character names using spaCy PERSON entities, unify them, assign round-robin speaker IDs."
    )
    parser.add_argument("text_file", help="Path to .txt file.")
    parser.add_argument("--synonyms", default="character_synonyms.json", help="Path to synonyms JSON.")
    parser.add_argument("--voice_map", default="character_voices.json", help="Path to voice map JSON.")
    args = parser.parse_args()

    identify_characters_in_text(
        text_file=args.text_file,
        synonyms_json=args.synonyms,
        voice_map_json=args.voice_map
    )
