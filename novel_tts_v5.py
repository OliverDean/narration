#!/usr/bin/env python3
"""
Advanced Novel TTS (Version 5)

Features:
  1) Full-document coreference resolution (Hugging Face).
  2) Nested quotes detection for multiple quotes in a paragraph.
  3) Character name unification (synonyms).
  4) Round-robin speaker assignment if the text has more unique speakers 
     than the TTS model supports.
  5) Dedicated "Narrator" voice:
     - Reads narration (text outside quotes).
     - Fills in for lines with unknown or excluded speakers (or you can skip them).
  6) Optionally exclude specific characters (skip or reassign them to narrator).
  7) Chunking long lines for TTS to avoid memory/time issues.

Usage Example:
  python advanced_novel_tts_v5.py path/to/story.txt \
      --model_name tts_models/en/vctk/vits \
      --gpu \
      --exclude_characters Lucy Verona Avery

If you have multiple files:
  python advanced_novel_tts_v5.py --batch chapter1.txt chapter2.txt ...
  
Dependencies:
  pip install TTS
  pip install spacy
  pip install transformers
  python -m spacy download en_core_web_trf

Version: 5
"""

import argparse
import os
import sys
import wave
import re
import json
import tempfile

import spacy
from transformers import pipeline
from TTS.api import TTS

##############################################################################
# GLOBAL CONFIG
##############################################################################

VOICE_MAP_FILE = "character_voices_v5.json"

# For demonstration, synonyms to unify references:
CHARACTER_SYNONYMS = {
    "Lucy": ["Lucy Ellingson", "Ms Lucy", "Lucy E."],
    "John": ["John Jr.", "Mr. Johnson"]
    # ... add more if desired
}

def unify_character_name(raw_name: str) -> str:
    """
    Normalize or unify names based on CHARACTER_SYNONYMS, 
    ensuring consistent naming for e.g. "Lucy Ellingson" => "Lucy".
    """
    normalized = raw_name.strip().title()
    for main_name, variants in CHARACTER_SYNONYMS.items():
        if normalized == main_name:
            return main_name
        for variant in variants:
            if variant.lower() in normalized.lower():
                return main_name
    return normalized

##############################################################################
# FILE STORAGE FOR VOICE MAP
##############################################################################

def load_voice_map() -> dict:
    """Load (or create) a global JSON map of character->speaker ID."""
    if os.path.isfile(VOICE_MAP_FILE):
        with open(VOICE_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_voice_map(voice_map: dict):
    """Save the updated voice map to JSON."""
    with open(VOICE_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(voice_map, f, indent=2)

##############################################################################
# WAV COMBINING
##############################################################################

def combine_wav_files(wav_files, output_file):
    """
    Combine multiple WAV files into a single WAV. 
    Ensures they share the same params.
    """
    if not wav_files:
        return

    with wave.open(wav_files[0], 'rb') as wf:
        params = wf.getparams()
        combined_data = wf.readframes(wf.getnframes())

    for wav_f in wav_files[1:]:
        with wave.open(wav_f, 'rb') as wf:
            if wf.getparams() != params:
                raise ValueError("WAV files differ in format; cannot combine.")
            combined_data += wf.readframes(wf.getnframes())

    with wave.open(output_file, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(combined_data)

##############################################################################
# TEXT CHUNKING FOR TTS
##############################################################################

def chunk_text_for_tts(text: str, max_chars: int = 200) -> list:
    """
    Break long lines into multiple segments, up to max_chars each.
    Attempts to break at a period if possible.
    """
    text = text.strip()
    segments = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
        start = end
    return segments

##############################################################################
# NESTED QUOTE DETECTION
##############################################################################

def find_dialogue_spans_nested(text: str):
    """
    Detect nested quotes in a paragraph using a stack approach.
    Returns a list of (quote_text, start_idx, end_idx).
    """
    OPEN_QUOTES = ["“", "‘", "\"", "'"]
    CLOSE_QUOTES = ["”", "’", "\"", "'"]

    results = []
    stack = []
    start_pos = None
    in_quote = False
    quote_char = None

    for i, ch in enumerate(text):
        if ch in OPEN_QUOTES:
            if not in_quote:
                in_quote = True
                start_pos = i + 1
                quote_char = ch
            else:
                # Nested quote
                stack.append((quote_char, start_pos))
                start_pos = i + 1
                quote_char = ch
        elif ch in CLOSE_QUOTES and in_quote:
            end_idx = i
            quote_text = text[start_pos:end_idx]
            results.append((quote_text, start_pos, end_idx))

            if stack:
                # Pop from stack for nested scenario
                prev_char, prev_start = stack.pop()
                start_pos = prev_start
                quote_char = prev_char
            else:
                # Quote fully closed
                in_quote = False
                quote_char = None
                start_pos = None
    return results

##############################################################################
# CORE REFERENCE & SPEAKER DETECTION
##############################################################################

def resolve_coref_text(coref_pipe, text: str) -> str:
    """
    Run a single pass of full-document coreference on 'text'.
    If pipeline fails or isn't provided, return original text.
    """
    if not coref_pipe:
        return text
    try:
        result = coref_pipe(text)
        return result.get("coref_resolved", text)
    except Exception as e:
        print(f"[WARN] Coreference resolution failed: {e}")
        return text

def find_speaker_in_context(context: str, known_speakers: list):
    """
    Look for direct "Name said" or "Name asked" patterns in 'context' 
    for any name in known_speakers.
    """
    speech_verbs = [
        "said", "asked", "replied", "muttered", "shouted", "yelled", 
        "whispered", "cried", "called", "snapped", "answered"
    ]
    for speaker in known_speakers:
        for verb in speech_verbs:
            pattern = rf"\b{speaker}\b\s+{verb}\b"
            if re.search(pattern, context, re.IGNORECASE):
                return speaker
    return None

def guess_speaker_for_quote(quote_text: str, 
                            doc_text: str, 
                            doc_spacy, 
                            start_idx: int, 
                            end_idx: int, 
                            known_speakers: list) -> str:
    """
    Attempt to guess the speaker near the snippet in doc_text (already coref-resolved).
    We'll gather ~100 chars around the snippet, check "Name said" patterns, 
    or see if spaCy recognized a PERSON entity near those offsets.
    """
    window = 100
    doc_len = len(doc_text)
    # Locate the snippet
    pos = doc_text.find(quote_text)
    if pos < 0:
        pos = doc_text.find(quote_text[:10])

    if pos < 0:
        return None

    start_ctx = max(0, pos - window)
    end_ctx = min(doc_len, pos + len(quote_text) + window)
    context_str = doc_text[start_ctx:end_ctx]

    # 1) Direct "Name said" detection
    direct_match = find_speaker_in_context(context_str, known_speakers)
    if direct_match:
        return direct_match

    # 2) Named entity approach in spaCy near snippet
    quote_start_token = None
    quote_end_token = None
    for token in doc_spacy:
        if token.idx <= start_idx < token.idx + len(token.text):
            quote_start_token = token.i
        if token.idx <= end_idx < token.idx + len(token.text):
            quote_end_token = token.i

    if quote_start_token is not None and quote_end_token is not None:
        local_range = 10
        t_start = max(0, quote_start_token - local_range)
        t_end = min(len(doc_spacy), quote_end_token + local_range)

        possible_names = set()
        for ent in doc_spacy.ents:
            if ent.label_ == "PERSON":
                if (t_start <= ent.start <= t_end) or (t_start <= ent.end <= t_end):
                    unified_name = unify_character_name(ent.text)
                    possible_names.add(unified_name)

        intersection = possible_names.intersection(set(known_speakers))
        if len(intersection) == 1:
            return intersection.pop()

    return None

##############################################################################
# MAIN SCRIPT (VERSION 5)
##############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Novel TTS v5: multi-speaker with narrator, synonyms, round-robin, coref, nested quotes."
    )
    parser.add_argument("input_file", help="Path to a text file, or use --batch for multiple.")
    parser.add_argument("--batch", nargs="*", 
                        help="Process multiple text files in a single run.")
    parser.add_argument("--model_name", default="tts_models/en/vctk/vits",
                        help="Coqui multi-speaker TTS model name.")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU if available.")
    parser.add_argument("--max_line_chars", type=int, default=200,
                        help="Max characters for TTS chunking.")
    parser.add_argument("--exclude_characters", nargs="*", default=["Lucy", "Verona", "Avery"],
                        help="Character names to exclude. Default: Lucy,Verona,Avery.")
    parser.add_argument("--skip_excluded", action="store_true",
                        help="If set, lines from excluded speakers are skipped entirely. Otherwise reassign to narrator.")
    args = parser.parse_args()

    # Gather input files
    if args.batch:
        input_files = args.batch
    else:
        input_files = [args.input_file]

    # 1) Load spaCy
    print("[INFO] Loading spaCy model (en_core_web_trf)...")
    try:
        nlp = spacy.load("en_core_web_trf")  # or "en_core_web_trf" if installed
    except Exception as e:
        print(f"[ERROR] Cannot load spaCy model: {e}")
        sys.exit(1)

    # 2) Load Hugging Face coref pipeline
    print("[INFO] Loading Hugging Face coref model: allenai/coref-spanbert-large...")
    try:
        coref_pipe = pipeline("coreference-resolution", model="allenai/coref-spanbert-large")
    except Exception as e:
        print(f"[WARN] Failed to load coref pipeline: {e}")
        coref_pipe = None

    # 3) Load TTS model
    print(f"[INFO] Loading TTS model: {args.model_name}")
    try:
        tts = TTS(model_name=args.model_name, gpu=args.gpu)
    except Exception as e:
        print(f"[ERROR] Failed to load TTS model: {e}")
        sys.exit(1)

    # 4) Verify multi-speaker support
    try:
        available_speakers = [s.strip() for s in tts.speakers if s.strip() != "ED"]
        if not available_speakers:
            raise ValueError("No speaker IDs in the model.")
    except AttributeError:
        print("[ERROR] This TTS model does not appear to support multiple speakers.")
        sys.exit(1)

    # 5) Load or create global voice map
    voice_map = load_voice_map()

    # 6) Round-robin approach for new speakers
    next_speaker_counter = 0
    def get_next_speaker_id():
        nonlocal next_speaker_counter
        assigned_id = available_speakers[next_speaker_counter % len(available_speakers)]
        next_speaker_counter += 1
        return assigned_id

    # 7) Ensure we have a "Narrator" voice in the map
    if "Narrator" not in voice_map:
        voice_map["Narrator"] = get_next_speaker_id()
        print(f"[INFO] Assigned Narrator -> {voice_map['Narrator']}")
    narrator_id = voice_map["Narrator"]

    def get_or_create_speaker_id(char_name: str):
        """Unify synonyms, then retrieve or create a speaker ID for that name."""
        unified = unify_character_name(char_name)
        if unified in voice_map:
            return voice_map[unified]
        else:
            new_id = get_next_speaker_id()
            voice_map[unified] = new_id
            return new_id

    # 8) Process each file
    for file_path in input_files:
        if not os.path.isfile(file_path):
            print(f"[ERROR] File not found: {file_path}")
            continue

        print(f"[INFO] Processing: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # (A) Full-document coreference
        resolved_text = resolve_coref_text(coref_pipe, raw_text)

        # (B) Split into paragraphs
        paragraphs = [p.strip() for p in resolved_text.split("\n\n") if p.strip()]

        # We'll also create one spaCy doc for the entire resolved text
        doc_spacy = nlp(resolved_text)

        # We'll accumulate all lines as (speaker_id, text)
        final_lines = []
        current_offset = 0

        for paragraph in paragraphs:
            # Find nested quotes
            spans = find_dialogue_spans_nested(paragraph)

            if not spans:
                # No quotes => entire paragraph read by narrator
                if paragraph.strip():
                    final_lines.append((narrator_id, paragraph.strip()))
                current_offset += len(paragraph) + 2
                continue

            # We do have quotes
            last_end = 0
            for (quote_text, q_start, q_end) in spans:
                # (1) Narration chunk before the quote
                narration_chunk = paragraph[last_end:q_start].strip()
                if narration_chunk:
                    final_lines.append((narrator_id, narration_chunk))

                # (2) Attempt to guess speaker
                speaker = guess_speaker_for_quote(
                    quote_text=quote_text,
                    doc_text=paragraph,
                    doc_spacy=doc_spacy,
                    start_idx=q_start + current_offset,
                    end_idx=q_end + current_offset,
                    known_speakers=list(voice_map.keys())  # We might also glean new ones from doc.ents
                )

                # If we found a speaker, unify & check exclusion
                if speaker is not None:
                    unified_name = unify_character_name(speaker)
                    if unified_name in args.exclude_characters:
                        # skip or reassign
                        if args.skip_excluded:
                            pass  # skip
                        else:
                            # reassign to narrator
                            if quote_text.strip():
                                final_lines.append((narrator_id, quote_text.strip()))
                    else:
                        speaker_id = get_or_create_speaker_id(unified_name)
                        if quote_text.strip():
                            final_lines.append((speaker_id, quote_text.strip()))
                else:
                    # Unknown => narrator
                    if quote_text.strip():
                        final_lines.append((narrator_id, quote_text.strip()))

                last_end = q_end

            # (3) Trailing narration
            if last_end < len(paragraph):
                trailing = paragraph[last_end:].strip()
                if trailing:
                    final_lines.append((narrator_id, trailing))

            current_offset += len(paragraph) + 2

        # (C) Generate TTS from final_lines
        if not final_lines:
            print("[INFO] No lines to read for this file.")
            continue

        temp_files = []
        try:
            for idx, (speaker_id, line_text) in enumerate(final_lines):
                chunks = chunk_text_for_tts(line_text, args.max_line_chars)
                for cid, chunk in enumerate(chunks):
                    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_wav_name = tmp_wav.name
                    tmp_wav.close()

                    try:
                        tts.tts_to_file(text=chunk, speaker=speaker_id, file_path=tmp_wav_name)
                        temp_files.append(tmp_wav_name)
                        short_preview = chunk[:60].replace("\n", " ")
                        if len(chunk) > 60:
                            short_preview += "..."
                        print(f"  [{idx+1}/{len(final_lines)}] Speaker='{speaker_id}' => \"{short_preview}\"")
                    except Exception as e:
                        print(f"[ERROR] TTS generation failed on line {idx+1}, chunk {cid+1}: {e}")
                        for ftemp in temp_files:
                            if os.path.exists(ftemp):
                                os.remove(ftemp)
                        sys.exit(1)

            # Combine partial WAV files
            base_name = os.path.splitext(file_path)[0]
            output_wav = base_name + "_v5.wav"
            print("[INFO] Combining partial audio segments...")
            combine_wav_files(temp_files, output_wav)
            print(f"[INFO] Audio file generated: {output_wav}")

        finally:
            for tf in temp_files:
                if os.path.exists(tf):
                    os.remove(tf)

        # Save updated voice map after each file
        save_voice_map(voice_map)

    print("[INFO] All done.")

if __name__ == "__main__":
    main()
