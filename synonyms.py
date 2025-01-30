#!/usr/bin/env python3
"""
Build or update a character_synonyms.json by scanning a text file for 'PERSON' entities
using spaCy, then fuzzy-match them against known canonical names from voice_map.

Key steps:
 1) Read full text.
 2) spaCy NER => find all PERSON entities (like "Lucy Ellingson", "Luccy", "Lucy’s").
 3) Clean them (remove trailing "'s", punctuation).
 4) Fuzzy-match them to existing canonical names from voice_map.json if any match is above threshold.
     -> If found match, store that entity as a synonym for canonical name.
     -> Otherwise, skip or treat as a new name if you want to update the voice_map, etc.
 5) Save synonyms in character_synonyms.json

You can tweak 'fuzzy_match_threshold' for how strict the matching should be.
By default, an 80% match is considered 'close enough' to unify to a known name.
"""

import os
import re
import json
import spacy
import argparse
from rapidfuzz.fuzz import ratio

##############################################################################
# Functions
##############################################################################

def load_json_if_exists(path):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def clean_entity_text(ent_text: str) -> str:
    """
    Remove trailing "'s" or "’s", some punctuation, extra spaces, etc.
    E.g. "Lucy’s" => "Lucy", "Lucy Ellingson." => "Lucy Ellingson"
    """
    # remove trailing punctuation
    txt = re.sub(r"[.,!?;:]+$", "", ent_text.strip()) 
    # remove trailing 's or ’s
    txt = re.sub(r"[’']s\b", "", txt, flags=re.IGNORECASE)
    return txt.strip()

def build_or_update_synonyms(
    text_file,
    voice_map_json="character_voices.json",
    synonyms_json="character_synonyms.json",
    fuzzy_match_threshold=80
):
    """
    1) Read text_file.
    2) spaCy => PERSON entities
    3) For each entity, clean it, fuzzy-match to known canonical names in voice_map.
    4) If above threshold, add to synonyms of that canonical name in synonyms_json.
    """

    # load text
    if not os.path.isfile(text_file):
        print(f"[ERROR] No such text file: {text_file}")
        return

    with open(text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # load voice_map => keys are canonical names, e.g. "Lucy", "Verona"...
    voice_map = load_json_if_exists(voice_map_json)
    # load synonyms => dict: { "Lucy": ["Lucy's", "Lucy E.", ...], ... }
    synonyms_dict = load_json_if_exists(synonyms_json)

    # Make sure synonyms_dict has an entry for each canonical name in voice_map
    for cname in voice_map.keys():
        if cname not in synonyms_dict:
            synonyms_dict[cname] = []

    # spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(raw_text)

    # gather newly found synonyms
    new_syn_count = 0

    # For each PERSON entity
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            cleaned = clean_entity_text(ent.text)
            if len(cleaned) < 2:
                continue

            # Fuzzy-match to existing canonical names
            # We'll keep track of the best match
            best_name = None
            best_score = 0

            for canonical_name in voice_map.keys():
                score = ratio(cleaned.lower(), canonical_name.lower())
                if score > best_score:
                    best_score = score
                    best_name = canonical_name

            # If best_score above threshold => treat as synonym of best_name
            if best_score >= fuzzy_match_threshold and best_name is not None:
                # add 'cleaned' to synonyms of best_name if not present
                if cleaned not in synonyms_dict[best_name]:
                    synonyms_dict[best_name].append(cleaned)
                    new_syn_count += 1
            else:
                # We could optionally do something if no match found,
                # e.g. treat it as a new name. But that's your choice.
                # For now, we skip unrecognized names.
                pass

    # Save updated synonyms
    if new_syn_count > 0:
        save_json(synonyms_dict, synonyms_json)
        print(f"[INFO] Found {new_syn_count} new synonyms. Updated {synonyms_json}.")
    else:
        print("[INFO] No new synonyms discovered.")


##############################################################################
# CLI
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Build advanced synonyms list using spaCy PERSON + fuzzy matching.")
    parser.add_argument("text_file", help="Path to the .txt file for scanning.")
    parser.add_argument("--voice_map", default="character_voices.json",
                        help="Path to voice map JSON with canonical names as keys.")
    parser.add_argument("--synonyms_out", default="character_synonyms.json",
                        help="Output synonyms JSON path.")
    parser.add_argument("--threshold", type=int, default=80,
                        help="Fuzzy match threshold (0-100). Default=80 means 80% similarity.")
    args = parser.parse_args()

    build_or_update_synonyms(
        text_file=args.text_file,
        voice_map_json=args.voice_map,
        synonyms_json=args.synonyms_out,
        fuzzy_match_threshold=args.threshold
    )

if __name__ == "__main__":
    main()
